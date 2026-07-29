# TRINITY Reference Physics Specification

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

**Status (2026-07-29):** 📘 reference — physics spec for the bugfix/code-audit review, built without reading the implementation.

---

## 0. How to read this document

### 0.1 Purpose and construction rule

This is the **yardstick** for an independent correctness audit of TRINITY. It states what the code
*should* compute, derived without reading `trinity/` (the source package). The author of this
document read only: `README.md`, `docs/source/*.rst`, `paper/` (figure scripts + captions),
`CLAUDE.md`, `param/*.param`, and `trinity/_input/default.param` (a config file, the single
permitted exception), plus external literature. **No implementation file was read and no code was
run.** Consequently every number here is either (i) from the literature, (ii) derived from first
principles in this document, or (iii) a parameter *declaration* from `default.param` /
`parameters.rst`. Where the code does something different, that is a finding — not a defect of this
spec — unless the spec is marked low-confidence.

### 0.2 Provenance labels

Every claim carries exactly one primary provenance label:

| Label | Meaning |
|---|---|
| **(a) TRINITY-claim** | Asserted by the TRINITY paper abstract, `README.md`, `docs/source/*.rst`, `paper/` scripts, or `default.param`/`registry` documentation strings. This is what the project *says* it does. |
| **(b) Literature** | Asserted by the external literature (Weaver+77, Rahner+17, Bonnor/Ebert, Spitzer, Cowie & McKee, Gnat & Ferland, …). |
| **(c) Derived** | Derived inside this document from first principles; the derivation is given or sketched. These are the strongest claims here because they do not depend on a source I could not fetch. |

Plus a confidence tag: **[HIGH]**, **[MED]**, **[LOW]**.

`AMBIGUOUS` marks a claim where the literature, the TRINITY docs, and/or first principles admit
more than one defensible reading. Both readings are given. **An `AMBIGUOUS` marker is itself an
audit finding**: the code has necessarily picked one, and the audit's job is to identify which, and
whether the pick is documented.

### 0.3 Source-access log (read this before trusting any citation)

The container's egress proxy **denied all outbound HTTPS except `github.com` /
`raw.githubusercontent.com`** (`CONNECT tunnel failed, response 403`). Verified against
`curl -sS "$HTTPS_PROXY/__agentproxy/status"`, which logged
`{"kind":"connect_rejected","host":"arxiv.org:443"}`.

| Source | Wanted | Got |
|---|---|---|
| arXiv:2605.27517 (TRINITY paper I) | full text, all equations | **FAILED** — abs/PDF/HTML all 403. Only the title, author list, and a paraphrased abstract recovered via `WebSearch`. |
| Weaver, McCray, Castor, Shapiro & Moore 1977, ApJ 218, 377 | Eq. numbers + prefactors | **FAILED** (ADS + articles.adsabs 403). Recovered via `WebSearch` snippets + own derivation. |
| Rahner, Pellegrini, Glover & Klessen 2017, MNRAS 470, 4453 | shell EOM, phase criteria | **FAILED** (OUP/arXiv 403). Abstract-level summary only, via `WebSearch`. |
| Rahner PhD thesis (imprs-hd.mpg.de) | bubble/shell structure detail | **FAILED** (403). |
| Bonnor 1956 / Ebert 1955 | ξ_crit, contrast, critical mass | **PARTIAL** — the three critical numbers confirmed by `WebSearch` snippets (ξ≈6.45, contrast 14.04, m=1.18); primary text not read. |
| Starburst99 docs (stsci.edu) | column definitions + units | **FAILED** (403). Search snippets only. |
| Gnat & Ferland 2012, ApJS 199, 20 | Λ(T) normalisation | **PARTIAL** — title/scope confirmed ("Ion-by-ion cooling efficiencies", 10⁴–10⁸ K, Cloudy v10, optically thin, low density); the normalisation convention (n_e n_H vs n_e n_ion vs n²) **not** confirmed. |
| El-Badry et al. 2019, MNRAS 490, 1961 (arXiv:1902.09547) | "Eq. 47" cited in `default.param` | **FAILED** — title/scope confirmed only; Eq. 47 itself never read. |

**Consequence:** every claim below whose only support is "(b) Literature" and which is *not*
independently re-derived here is at most **[MED]** confidence, and any bare equation *number* from
Weaver+77 is **[LOW]** — I never opened the paper. Where the task asked me to "say which equation
number each really is", the honest answer is recorded in §3.6: I could not verify Weaver's equation
numbering, so I refuse to assert it.

---

## 1. What the code is for

### SPEC-001 — Scientific question
**Claim.** TRINITY answers: *for a single, isolated, spherically symmetric giant molecular cloud
hosting one coeval star cluster, does pre-supernova stellar feedback disperse the cloud, does the
cloud re-collapse, and how much ionizing radiation escapes?* It is a 1-D thin-shell semi-analytic
model, not a hydrodynamics code.
**Source.** (a) TRINITY-claim — paper title *"TRINITY: A coupled model of winds, radiation, and
photoionised gas in molecular clouds. I. Methods and validation"* (Teh, Klessen, Glover, Kreckel
2026); `README.md`; `docs/source/index.rst` ("resolves the phase transitions … and stopping fate").
**Regime.** Single cluster, single cloud, spherical symmetry, thin shell. **[HIGH]**

### SPEC-002 — Physical picture (the model the code must implement)
**Claim.** Four concentric regions, ordered outward, following Weaver+77's four-zone structure:
1. **Free-streaming wind**, `r < R1`: cluster wind at terminal speed `v_w`, `ρ_w = Ṁ_w/(4πr²v_w)`.
2. **Shocked wind / hot bubble**, `R1 < r < R2`: `T ~ 10⁶–10⁷ K`, near-isobaric at pressure `P_b`.
   `R2` is the contact discontinuity.
3. **Swept-up shell**, `R2 < r < R_sh,out`: cold/warm dense gas, geometrically thin, containing an
   inner photoionized (`~10⁴ K`) layer and an outer neutral/molecular (`~10²  K`) layer.
4. **Undisturbed cloud, then ambient ISM**, `r > R_sh,out`.
**Source.** (b) Literature (Weaver+77 zone structure) + (a) TRINITY-claim (`docs/source/running.rst`
snapshot keys `R1`, `R2`, `Pb`, `T0`; `paper_teaser.py` plots `R_ts` (wind termination shock),
`R_b`, `R_sh`). **[HIGH]**

### SPEC-003 — Inputs
**Claim.** The physical inputs and their declared units are:

| Input | Symbol | Unit | Default | Meaning |
|---|---|---|---|---|
| `mCloud` | `M_cl` | M⊙ | 1e7 | total GMC mass |
| `sfe` | `ε` | – | 0.01 | star-formation efficiency, `0 < ε < 1` |
| `ZCloud` | `Z` | Z⊙ | 1 | metallicity (**only 1.0 supported**) |
| `nCore` | `n_core` | cm⁻³ | 1e5 | *hydrogen-nuclei* number density of the core |
| `rCore` | `r_core` | pc | 0.01 | core radius (unused if `densPL_alpha = 0`) |
| `nISM` | `n_ISM` | cm⁻³ | 1 | ambient hydrogen-nuclei density |
| `PISM` | `P_ISM/k_B` | K cm⁻³ | 0 | ambient thermal pressure |
| `dens_profile` | – | – | `densPL` | `densPL` (power law) or `densBE` (Bonnor–Ebert) |
| `densPL_alpha` | `α` | – | 0 | power-law index, `-2 ≤ α ≤ 0` |
| `densBE_Omega` | `Ω` | – | 14.1 | `ρ_core/ρ_edge` for Bonnor–Ebert |

**Source.** (a) TRINITY-claim — `trinity/_input/default.param`, `docs/source/parameters.rst`. **[HIGH]**

### SPEC-004 — Derived cluster and gas masses
**Claim.** `M_cluster = ε · M_cl` and `M_cloud,after = M_cl − M_cluster = (1−ε) M_cl`.
**Source.** (a) TRINITY-claim — `docs/source/parameters.rst`, *Derived quantities*. **[HIGH]**

### SPEC-005 — `AMBIGUOUS`: which mass normalises the cloud density profile
**Reading A.** `r_cloud` is solved from the **post-SFE gas mass** `(1−ε)M_cl` at fixed `n_core`
(mass conservation: stars form out of the cloud, the remaining gas is more compact).
**Reading B.** `r_cloud` is solved from the **total** `M_cl`, and the cluster mass is removed only
from the gravitating/sweepable budget.
**Evidence for A.** `paper/methods/figures/paper_densityProfile.py` `_DEFAULTS` sets
`mCloud=1e5 * (1 - 0.01)` with the comment `# Msun, post-SFE`, and feeds it to
`compute_rCloud_*`. That is a figure script, not the solver, but it is written by the same author.
**Why it matters.** At `ε = 0.3` (`param/simple_cluster.param`!) the two readings differ in
`r_cloud` by `(0.7)^{1/3} = 0.888`, i.e. 11% in radius and 30% in swept mass — far above any
numerical tolerance.
**Source.** (c) Derived contradiction; (a) partial evidence. **[MED]** → **audit must determine
which the solver uses and whether it is stated anywhere.**

### SPEC-006 — Outputs
**Claim.** Per-snapshot state written to `dictionary.jsonl` must include at least: `t_now`, `R2`,
`v2`, `Eb`, `T0`, `R1`, `Pb`, `current_phase`, `SimulationEndReason`, the SPS drivers (`Lmech_W`,
`Lmech_SN`, `Qi`, `Lbol`, `pdot_total`), the pressures (`P_drive`, `P_HII`, `P_ram`), the forces
(`F_grav`, `F_ram`, `F_ram_wind`, `F_ram_SN`, `F_HII`, `F_rad`), and downsampled bubble/shell
profiles. Run constants and the termination block live in `metadata.json`.
**Source.** (a) TRINITY-claim — `docs/source/running.rst`, *Output data model*. **[HIGH]**

### SPEC-007 — Force-budget closure (auditable invariant)
**Claim.** The recorded forces must close the momentum equation: at every snapshot,
`M_sh dv2/dt = F_drive + F_rad − F_grav − F_ram,ambient − 4πR2²P_ISM` to within integrator
tolerance, where `F_drive = 4πR2² P_drive`. A stacked force-*fraction* plot (which
`paper_feedback.py` and `paper_teaser.py` both produce, normalising to `F_tot`) is only meaningful
if the listed terms are exhaustive and non-overlapping.
**Source.** (c) Derived — the fractions in `paper_feedback.py` are normalised by `F/F_tot`, which
presupposes closure. **[HIGH]** — *this is the single cheapest global audit test.*

---

## 2. Phase structure

### SPEC-010 — The phase sequence
**Claim.** Phases, in order, with the literal `current_phase` strings the outputs use:
`energy` → `implicit` → `transition` → `momentum`, terminating in one of the stopping fates of §9.
`implicit` is documented as *"a numerical continuation of the energy phase"* and is merged into
`energy` for display.
**Source.** (a) TRINITY-claim — `docs/source/index.rst` ("energy-driven → transition →
momentum-driven"); `paper_teaser.py` `_DISPLAY_PHASE_MAP = {"implicit": "energy"}`;
`paper_densityProfile.py` `map_phase` maps `('energy','implicit') → 'energy'`. **[HIGH]**

### SPEC-011 — Energy-driven phase: physical definition
**Claim.** The shocked-wind gas retains its thermal energy on the expansion time
(`t_cool,bubble ≫ t_dyn = R2/v2`). The bubble is a hot, near-isobaric reservoir and the shell is
pushed by `4πR2²P_b`, with `P_b` far exceeding the bare wind momentum flux
(`4πR2²P_b ≫ ṗ_w`). Energy, not momentum, is the conserved driver, and the bubble does `PdV` work
on the shell.
**Source.** (b) Literature (Weaver+77 §II–III); (c) Derived (the definition of the adiabatic limit).
**Regime.** `t_cool ≫ t_dyn`, `R1 ≪ R2`. **[HIGH]**

### SPEC-012 — Momentum-driven phase: physical definition
**Claim.** The shocked-wind gas has radiated away (or vented) its thermal energy; the bubble no
longer supplies pressure support, and the wind/SN ejecta deposit only their **momentum** at the
shell. The driving term degenerates to `ṗ_w + ṗ_SN` (plus radiation and, in TRINITY, `P_HII`).
**Source.** (b) Literature (Rahner+17 WARPFIELD; classical momentum-driven/"snowplough" limit);
(a) TRINITY-claim (paper abstract: momentum phase driver = "photoionised-gas pressure plus direct
ram pressure"). **[HIGH]**

### SPEC-013 — The *correct* physical criterion for energy → momentum
**Claim.** The physically meaningful criterion is that the hot bubble stops being an energy
reservoir, i.e. the radiative (plus mixing/leakage) loss rate approaches the mechanical input rate:

```
    L_loss / L_gain → 1        equivalently   (L_gain − L_loss)/L_gain → 0
```

with `L_gain = η_th L_mech` and `L_loss = L_cool,bubble + L_conduction-front + L_leak`. Equivalent
formulations in the literature are `t_cool,bubble < t_dyn` and "the bubble's cooling radius has been
reached".
**Source.** (b) Literature (standard superbubble criterion, e.g. Mac Low & McCray 1988 style);
(a) TRINITY-claim — `phaseSwitch_LlossLgain = 0.05` is *exactly* this criterion with a 5% floor.
**[HIGH]** for the physics, **[MED]** for the mapping to the parameter.

### SPEC-014 — `AMBIGUOUS`: the transition criterion has no unique value in the literature
**Reading A (`cooling_balance`, the TRINITY default).** Fire when
`(L_gain − L_loss)/L_gain ≤ 0.05`. **The 0.05 is a numerical regularisation, not physics** — the
physical statement is "→ 0", and any threshold in `0.01…0.2` is equally defensible. The predicted
transition *time* is therefore threshold-dependent, and the audit should quantify that sensitivity.
**Reading B (`ebpeak`).** Fire when the net bubble energy rate turns over, `Ė_b ≤ 0`. This is a
strictly later and better-posed event (the bubble has actually started losing energy) and is
threshold-free.
**Reading C (`blowout`).** Fire when `R2 > r_cloud`. This is a *geometric* event, not a thermal one;
it is a proxy for "the bubble vents and depressurises", which is physically real but is a different
mechanism from radiative collapse.
**Source.** (a) TRINITY-claim — `transition_trigger` accepts `cooling_balance`, `blowout`,
`ebpeak`, and the alias `r1 = blowout,ebpeak`, "fires on whichever occurs first". **[HIGH]** that
all three exist; **[HIGH]** that they are physically distinct events.

### SPEC-015 — `AMBIGUOUS`: 1-D conduction under-predicts bubble energy loss
**Claim.** Modern 3-D work (El-Badry et al. 2019, MNRAS 490, 1961; Lancaster et al. 2021a,b,
"Efficiently cooled stellar wind bubbles in turbulent clouds" I & II) finds that turbulent mixing
across a *fractal* contact discontinuity removes bubble energy far faster than 1-D Spitzer
conduction predicts, so a 1-D Weaver bubble stays energy-driven **much longer than reality**. Any
1-D code that reproduces Weaver exactly will therefore over-predict the energy phase duration.
**Consequence for the audit.** TRINITY exposes four separate ad-hoc knobs that patch precisely this
(`cooling_boost_mode`, `cooling_boost_fmix`, `cooling_boost_theta`, `cooling_boost_kappa`,
`cooling_boost_fA`), and `param/paperII_grid_sweep.param` *ships with*
`cooling_boost_mode multiplier` / `cooling_boost_fmix 4`. **Published Paper-II results are
therefore NOT the byte-identical default physics.** The audit must not treat the default
(`cooling_boost_mode none`) run as "the published model".
**Source.** (b) Literature; (a) TRINITY-claim (`default.param` knob docstrings; the sweep file).
**[HIGH]**

### SPEC-016 — Transition phase: what it must be
**Claim.** A physically motivated *transition* phase exists because the switch is not instantaneous:
the bubble pressure decays over a finite time as it radiates, and the driving term must migrate
continuously from `4πR2²P_b` to `ṗ_w + ṗ_SN`. Rahner+17's original WARPFIELD assumed the switch was
instantaneous; that is explicitly described in the literature as a simplification. A correct
implementation must be continuous in `dv2/dt` across the handover, or explicitly document the
discontinuity.
**Source.** (b) Literature (WARPFIELD's instantaneous-transition simplification is stated in the
WARPFIELD 2.0 description); (c) Derived (continuity requirement).
**Audit test.** Sample `dv2/dt` on either side of each phase boundary; a jump > integrator
tolerance is a finding. **[MED]**

### SPEC-017 — Re-collapse is a *fate*, not a phase
**Claim.** `collapse` appears as a fourth bar in the published phase-timeline figure, but it is
constructed in post-processing as "the part of the momentum phase after `v2` crosses zero from
positive to negative" — it is not a separate integrator phase.
**Source.** (a) TRINITY-claim — `paper_densityProfile.py::_extract_phase_info` splits the final
`momentum` interval at the interpolated `v2 = 0` crossing and labels the remainder `collapse`.
**[HIGH]**

---

## 3. Governing equations

### 3.1 Shell equation of motion

### SPEC-020 — Thin-shell momentum equation (canonical form)
**Claim.** The shell equation of motion is the momentum equation for a thin spherical shell of mass
`M_sh(R2)` sweeping up a static ambient medium:

```
    d/dt [ M_sh v2 ]  =  4πR2² ( P_drive − P_ext )  +  F_rad  −  F_grav
```

with `Ṁ_sh = 4πR2² ρ_amb(R2) v2` while the shell expands into fresh gas.
**Equivalent, expanded, form** (the one most codes integrate):

```
    M_sh dv2/dt = 4πR2² ( P_drive − P_ext − ρ_amb(R2) v2² ) + F_rad − F_grav
```

**Symbols.** `M_sh` [M⊙] swept mass; `v2 = dR2/dt` [pc Myr⁻¹]; `R2` [pc]; `P_drive`, `P_ext`
[M⊙ pc⁻¹ Myr⁻² ≡ dyn cm⁻²]; `ρ_amb` [M⊙ pc⁻³]; `F_rad`, `F_grav` [M⊙ pc Myr⁻²].
**Source.** (c) Derived (Newton's second law for a variable-mass shell accreting material at rest);
(b) Literature (standard WARPFIELD form).
**AUDIT TRAP (classic bug class).** The `−4πR2²ρ_amb v2²` "ram pressure" term is **already
contained** in `d(M_sh v2)/dt` via `Ṁ_sh v2`. Writing `M_sh dv2/dt = … − 4πR2²ρ_amb v2²` *and*
`d(M_sh v2)/dt = …` in different code paths, or adding the ram term to the first form a second
time, double-counts it. The snapshot key `F_ram` in `docs/source/running.rst` sits next to
`F_ram_wind`/`F_ram_SN`, which suggests `F_ram` there means *feedback* ram pressure, not the
sweep-up term — the audit must confirm the sweep-up term appears exactly once. **[HIGH]**

### SPEC-021 — Swept-up shell mass
**Claim.** `M_sh(R2) = ∫₀^{R2} 4πr² ρ_cloud(r) dr = M_enc(R2)` while `R2 ≤ r_cloud`, and
`M_sh(R2) = M_cloud,after + (4π/3)ρ_ISM (R2³ − r_cloud³)` beyond the cloud edge. All cloud gas
interior to `R2` is in the shell (there is nowhere else for it to be, given the hot bubble is
wind material plus evaporated shell gas).
**Source.** (c) Derived; (a) TRINITY-claim (`M_enc(<r)` is the plotted "ingredient" in
`paper_densityProfile.py`). **[HIGH]**

### SPEC-022 — Driving pressure: TRINITY's central novelty (`P_drive`)
**Claim.** TRINITY's stated departure from WARPFIELD is a **phase-aware** driving pressure:

```
    energy / implicit phase :   P_drive = max( P_b , P_HII )
    transition phase        :   P_drive = max( P_b , P_HII + P_ram )
    momentum phase          :   P_drive =      P_HII + P_ram
```

with `P_ram = ( ṗ_w + ṗ_SN ) / (4πR2²)`.
**Source.** (a) TRINITY-claim — paper abstract, recovered verbatim-in-substance via search: *"A
phase-aware driving prescription uses the larger of the hot-bubble and photoionised-gas pressures
in the energy-driven phase, and photoionised-gas pressure plus direct ram pressure in the
momentum-driven phase."* Corroborated independently by `paper/methods/figures/paper_feedback.py`,
whose module docstring says: *"Energy phase: … which branch of `max(Pb, P_HII)` is active"* and
*"Transition phase: … only when the non-bubble branch wins `max(Pb, P_HII + P_ram)`"*.
**[HIGH]** that this is the intended prescription.

### SPEC-023 — `AMBIGUOUS`: `max()` is a modelling choice, not a derived result
**Reading A (`max`, TRINITY's choice).** The shell's inner face is set by whichever reservoir is
over-pressured; the other is compressed away. Defensible as a limiting-case interpolation.
**Reading B (sum).** If the hot bubble and the photoionized layer are *radially stacked* (bubble
inside, HII layer between bubble and shell), the shell sees only the innermost adjacent pressure,
which is `P_HII` — not `max`. If they are *not* stacked, the pressures do not simply add either.
**Reading C (two-zone interface).** Solve for the actual radius where the ionization front sits and
apply the local pressure there; this is the physically correct treatment and is what a shell-
structure integration should give for free.
**Consequences the audit must check.** (i) `max(·)` is **not differentiable**: the RHS of the ODE
has a kink whenever the two pressures cross, which an adaptive stiff solver (LSODA) will chatter
on — `paper/methods/data/app_LSODA.npz` and the `rcloud_smoothing` figure suggest exactly this
class of problem was hit for `r_cloud`. (ii) `max()` is not conservative: work done on the shell is
no longer `∫P_b dV`, so the bubble energy equation's `PdV` term and the shell's work term can
disagree — a genuine energy-conservation leak. **[HIGH]** that this is a real ambiguity.

### SPEC-024 — Bubble thermal pressure
**Claim.** For an ideal gas with `γ = 5/3` filling the shell between `R1` and `R2`:

```
    E_b = P_b V_b /(γ−1) = (3/2) P_b V_b ,   V_b = (4π/3)(R2³ − R1³)
    ⇒   P_b = E_b / [ 2π (R2³ − R1³) ]
```

**Symbols.** `E_b` [M⊙ pc² Myr⁻²]; `V_b` [pc³]; `γ_adia = 5/3` (`default.param`).
**Source.** (c) Derived. **AUDIT TRAP:** using `V_b = (4π/3)R2³` (dropping `R1`) is Weaver's own
approximation and is fine when `R1 ≪ R2`, but `R1/R2` is *not* small in the momentum-approach
regime, where `P_b` collapses and `R1 → R2`. Check which volume the code uses and whether it is
consistent between `P_b(E_b)` and the `PdV` term. **[HIGH]**

### SPEC-025 — Wind termination shock radius `R1`
**Claim.** `R1` is set by ram-pressure balance between the free wind and the hot bubble:

```
    ρ_w(R1) v_w²  =  P_b        with  ρ_w(r) = Ṁ_w /(4π r² v_w)
    ⇒   R1 = sqrt( Ṁ_w v_w /(4π P_b) ) = sqrt( ṗ_w /(4π P_b) )
```

**`AMBIGUOUS`.** Strictly, the *post*-shock pressure of a strong shock with `γ=5/3` is
`(2/(γ+1))ρv² = (3/4)ρ_w v_w²`, giving `R1 = sqrt(3 ṗ_w /(16π P_b))`, smaller by `√3/2 = 0.866`.
Weaver+77 and most semi-analytic descendants drop the 3/4. Either is defensible; the audit should
record which, since `R1` enters `V_b` (SPEC-024) and hence `P_b`.
**Source.** (c) Derived; (b) Literature for the convention. **[HIGH]** for the balance,
**[HIGH]** for the ambiguity.

### 3.2 Radiation terms

### SPEC-026 — Direct radiation pressure
**Claim.**

```
    F_rad,dir = ( L_bol / c ) · f_abs ,      f_abs = 1 − exp(−τ_UV) ∈ [0,1]
```

`f_abs` is the fraction of the cluster's bolometric (UV/optical) luminosity absorbed by the shell
(gas + dust). In the optically thick limit `F_rad,dir → L_bol/c` — the single-scattering limit.
**Symbols.** `L_bol` [erg s⁻¹ → M⊙ pc² Myr⁻³]; `c = 2.99792458e10 cm s⁻¹` (`c_light` in
`default.param`); `τ_UV` dimensionless.
**Source.** (c) Derived (photon momentum flux `L/c`); (b) Literature (standard).
**Numeric anchor.** For `L_bol = 10⁴³ erg s⁻¹`, `L_bol/c = 3.3×10³² dyn`. **[HIGH]**

### SPEC-027 — Dust-reprocessed (IR) radiation pressure
**Claim.**

```
    F_rad,IR = τ_IR · L_bol / c ,     τ_IR = κ_IR Σ_sh = κ_IR M_sh /(4π R2²)
    total :  F_rad = (L_bol/c) ( 1 − e^{−τ_UV} + τ_IR )
```

**Symbols.** `κ_IR = 4 cm² g⁻¹` (`dust_KappaIR`, Rosseland mean per gram of **gas**, at Z⊙);
`Σ_sh` [g cm⁻²].
**Validity.** The `τ_IR` trapping factor is a single-scattering-times-optical-depth estimate valid
for `τ_IR ≲ few`. For `τ_IR ≫ 1` radiation-hydro simulations show the true boost saturates well
below `τ_IR` (leakage through low-column channels). For `τ_IR ≪ 1` the term must vanish.
**Source.** (b) Literature (standard IR-trapping prescription); (a) TRINITY-claim (`dust_KappaIR`
declared with unit `cm²/g`).
**AUDIT TRAP.** `κ_IR` must multiply the *mass* column `M_sh/(4πR2²)` in g cm⁻², **not** a number
column. And it should scale with metallicity (dust-to-gas ∝ Z); `default.param` gives `dust_noZ =
0.05 Z⊙` as the floor "below which there is effectively no dust", implying a Z scaling exists for
`dust_sigma` — check whether `dust_KappaIR` is scaled too, or is (inconsistently) fixed. **[MED]**

### SPEC-028 — Dust absorption of ionizing photons
**Claim.** Dust competes with hydrogen for Lyman-continuum photons with a per-hydrogen-nucleus
cross-section `σ_d = 1.5×10⁻²¹ cm² (Z/Z⊙)`; `τ_d = σ_d N_H`. The ionizing-photon budget must close:

```
    1 = f_ion,gas + f_ion,dust + f_esc      at every timestep
```

**Source.** (a) TRINITY-claim — `dust_sigma = 1.5e-21 cm²` in `default.param`; and
`paper_teaser.py` panel (c) is explicitly *"stacked area showing the fraction of ionising photons
absorbed by gas inside the shell, by dust inside the shell, and escaping past the shell, **summing
to unity at every timestep**"*.
**Audit test.** This is an exactly-checkable invariant on the published outputs. **[HIGH]**

### 3.3 Photoionized-gas pressure

### SPEC-029 — Strömgren balance and `P_HII`
**Claim.** In ionization equilibrium, recombinations balance the ionizing photon output:

```
    Q_i = (4π/3) α_B χ_e n_H² ( R_i³ − R_in³ )              (filled sphere between R_in and R_i)
    P_HII = ( 1 + x_He + χ_e ) n_H k_B T_ion  ≡  n_tot k_B T_ion
```

**Symbols.** `Q_i` [s⁻¹ → Myr⁻¹]; `α_B = 2.59×10⁻¹³ cm³ s⁻¹` (`caseB_alpha`, case B at 10⁴ K);
`χ_e = n_e/n_H = 1 + Z_He,shell·x_He = 1.1` (`chi_e_shell`, `default.param`);
`x_He = 0.1`; `T_ion = 10⁴ K` (`TShell_ion`). With `x_He = 0.1`, `Z_He,shell = 1`:
`n_tot = 2.2 n_H`, so `P_HII = 2.2 n_H k_B (10⁴ K)`.
**Cross-check.** `μ_ion,shell = 14/22` (declared in `parameters.rst`) gives
`n_tot = ρ/(μ_ion,shell m_H) = 1.4 n_H/(14/22) = 2.2 n_H` ✓ — the composition constants are
internally consistent.
**Source.** (c) Derived; (a) TRINITY-claim for the constants. **[HIGH]**

### SPEC-030 — `AMBIGUOUS`: what geometry sets `P_HII`
`default.param` says only *"HII pressure (from Strömgren ionization balance in shell)"*.
**Reading A.** The ionized gas is a *layer inside the shell* (between `R2` and the ionization front
within the shell). Then `n_H` in SPEC-029 is the *shell* density, which is high, so `P_HII` is
large and rises as the shell compresses — and the ionization balance is over a thin slab, not a
sphere.
**Reading B.** The ionized gas *fills the bubble-to-shell gap*, i.e. it is the classical HII region
interior to `R2`. Then `n_H` is a much lower, dynamically-set density.
These give qualitatively different `P_HII(t)`, and TRINITY's central claim rests on this term.
**Audit priority: HIGHEST.** **[HIGH]** that this is under-specified in every source I may read.

### 3.4 Gravity

### SPEC-031 — Gravitational force on the shell
**Claim.**

```
    F_grav = G M_sh ( M_cluster + M_sh/2 ) / R2²
```

**Derivation (c).** (i) The cluster is a point mass at the centre → `G M_cluster M_sh / R2²`.
(ii) The shell's self-gravity: the self-potential energy of a thin uniform shell of mass `M_sh` at
radius `R` is `U = −G M_sh²/(2R)`, so the inward self-force is `−dU/dR = −G M_sh²/(2R²)`; hence the
`M_sh/2` factor. (iii) Cloud gas **outside** `R2` exerts zero net force on the shell (Newton's shell
theorem) and must **not** be included.
**Symbols.** `G = 6.6743×10⁻⁸ cm³ g⁻¹ s⁻²` = `4.4985×10⁻³ pc³ M⊙⁻¹ Myr⁻²` (see SPEC-072).
**Source.** (c) Derived; (b) Literature (WARPFIELD uses the same `M_sh/2` form).
**AUDIT TRAPS.** (1) `M_sh/2` vs `M_sh` is a factor-2 error in the self-gravity term — the single
most common bug in shell models. (2) Once `R2 > r_cloud`, `M_sh` keeps growing from the ISM; the
formula still holds. (3) If any gas remains *interior* to `R2` and outside the shell (it should
not, SPEC-021), it must be added to the enclosed mass. **[HIGH]**

### SPEC-032 — Gravitational stall / escape criterion
**Claim.** The shell escapes (never re-collapses) iff its kinetic energy exceeds the binding energy
at all later times; the instantaneous escape speed is

```
    v_esc(R2) = sqrt( 2G ( M_cluster + M_sh ) / R2 )
```

**Physically meaningful stall:** `v2 → 0` with `dv2/dt < 0` while `F_grav` exceeds the sum of the
outward terms. **Re-collapse:** `v2 < 0` sustained.
**Source.** (c) Derived. **[HIGH]**

### 3.5 Bubble energy equation

### SPEC-035 — The bubble energy equation
**Claim.**

```
    dE_b/dt = L_gain − L_loss
    L_gain  = η_w L_mech,w + η_SN L_mech,SN
    L_loss  = P_b dV_b/dt  +  L_cool  +  L_leak
```

with `dV_b/dt = 4π( R2² v2 − R1² dR1/dt )`.
**Symbols.** `η_w = FB_thermCoeffWind = 1`, `η_SN = FB_thermCoeffSN = 1` (thermalisation
efficiencies, `default.param`); `L_cool` = radiative losses integrated over the bubble interior
*and* the conduction front; `L_leak` = venting loss (SPEC-036).
**Source.** (c) Derived (first law for the bubble as an open control volume); (a) TRINITY-claim for
the coefficient names.
**AUDIT TRAPS.** (i) The `P_b dV_b/dt` term must use the *same* `V_b` as `P_b = E_b/[2π(R2³−R1³)]`
(SPEC-024) — a mismatch is an energy leak. (ii) The work the bubble does on the shell must equal
the work the shell receives; if `P_drive = max(P_b, P_HII)` and `P_HII` wins (SPEC-023), the shell
receives `4πR2²P_HII v2` but the bubble still loses `4πR2²P_b v2` — energy is created or destroyed.
**This is a first-class audit item.** **[HIGH]**

### SPEC-036 — Covering-fraction (venting) loss
**Claim.** `default.param` defines `coverFraction = C_f` as the closed fraction of the bubble wall;
hot gas vents through area `(1−C_f)·4πR2²` at the interior sound speed. The energy flux through a
freely-venting area for a `γ = 5/3` gas is the **enthalpy** flux:

```
    L_leak = (1−C_f) · 4πR2² · c_s · [ γ/(γ−1) ] P_b  =  (1−C_f) · 4πR2² · c_s · (5/2) P_b
    with c_s = sqrt( γ P_b / ρ_b )
```

**`AMBIGUOUS`.** If instead the code advects only the internal energy density
`u = (3/2)P_b`, the leak is 40% smaller (`5/2` vs `3/2`). `default.param` says only "draining bubble
energy", so both readings are open. Note `C_f = 1.0` is the default and *"recovers the sealed
(Weaver) bubble exactly"*, so this term is off in the fiducial runs — but it is on in any run that
sets `C_f < 1`.
**Source.** (a) TRINITY-claim for the mechanism; (c) Derived for the flux. **[MED]**

### 3.6 Bubble interior structure (Weaver)

### SPEC-040 — Weaver interior similarity profiles
**Claim.** In the shocked-wind region, with Spitzer conduction and a conduction front at the
contact discontinuity `R2`, the self-similar solution gives, with `x ≡ r/R2`:

```
    T(r) = T_b · (1 − x)^{2/5}
    n(r) = n_b · (1 − x)^{−2/5}       (so that P_b = n_tot k_B T is r-independent: isobaric)
```

**Source.** (b) Literature (Weaver+77, the interior structure result reproduced in essentially
every superbubble paper since); the `2/5` and `−2/5` exponents were confirmed by search snippets.
**Regime.** `R1 ≪ r < R2`, conduction-dominated, radiative losses a perturbation.
**AUDIT NOTE.** `n(r)` diverges as `x → 1`; the code must integrate to `x = ξ_max < 1`. TRINITY's
`bubble_xi_Tb = 0.98` (*"the relative radius ξ = r/R2 at which we measure the bubble
temperature"*) is exactly such a cut. At `ξ = 0.98`, `(1−ξ)^{2/5} = 0.02^{0.4} = 0.209`, so the
"bubble temperature" `T0` reported at `ξ = 0.98` is **≈21% of the central `T_b`** — a factor of ~5.
Any comparison of the reported `T0` against a Weaver `T_b` formula must apply this factor.
**[HIGH]** for the profile shape, **[HIGH]** for the `ξ = 0.98` consequence.

### SPEC-041 — The (α, β, δ) similarity parameters
**Claim.** They are *logarithmic time derivatives* of the dynamical state, not absolute rates:

```
    α ≡ d ln R2 / d ln t = v2 t / R2
    β ≡ − d ln P_b / d ln t
    δ ≡   d ln T   / d ln t
```

**Source.** (a) TRINITY-claim — `default.param`: `cool_alpha 0.6` *"alpha = v2*t_now/R2"*,
`cool_beta 0.8` *"beta = - dPb/dt"*, `cool_delta -6/35` *"delta = dT/dt"*.
**Derived cross-check (c).** In the Weaver limit `α = 3/5`, `P_b ∝ t^{−4/5}` ⇒ `β = 4/5` ✓,
`T ∝ t^{−6/35}` ⇒ `δ = −6/35` ✓. **The TRINITY defaults are exactly the Weaver self-similar
exponents.** Isobaricity then forces `d ln n/d ln t = −(β + δ) = −4/5 + 6/35 = −22/35` ✓,
matching the Weaver interior density scaling. **[HIGH]**

### SPEC-042 — The conduction closure (derived, and it reproduces `δ = −6/35` exactly)
**Claim.** Balancing the conductive heat flux `C T^{7/2}/R2` against the isobaric enthalpy /
expansion terms `~ P_b R2² / t` fixes the interior temperature scale:

```
    T_b^{7/2} = a · P_b R2² / ( C t )          with a an O(1) pure number
```

Dimensional check: `[C T^{7/2}] = erg s⁻¹ cm⁻¹` and `[P R²/t] = erg cm⁻¹ s⁻¹` ✓ — no `k_B/μ`
factor is needed, because the `k/(μ m_H)` in the enthalpy flux cancels against the isobaric
`ρ ∝ 1/T`.
**Taking `d ln/d ln t` of the closure:**

```
    δ = (2/7) ( 2α − β − 1 )
```

and substituting the Weaver values `α = 3/5`, `β = 4/5`:
`δ = (2/7)(1.2 − 0.8 − 1) = (2/7)(−0.6) = −6/35` **exactly**.
**Source.** (c) Derived here, end to end. This is the **strongest testable structural claim in this
document**: it links three otherwise independent `default.param` constants and must hold for any
`(α, β, δ)` triple the implicit solver returns in the Weaver regime.
**Audit test.** Extract `(α, β, δ)` from a run's snapshots and check `δ ≈ (2/7)(2α − β − 1)` holds
during the energy phase, degrading only as cooling becomes important. **[HIGH]**

### SPEC-043 — Thermal conduction coefficient
**Claim.** Spitzer conductivity `κ(T) = C T^{5/2}` with heat flux `q = −C T^{5/2} ∇T`, and
`C = 6×10⁻⁷ erg s⁻¹ cm⁻¹ K⁻⁷ᐟ²`, which is Spitzer's `1.84×10⁻⁵/ln Λ` at `ln Λ ≈ 30`.
**Source.** (b) Literature (Spitzer 1962); (a) TRINITY-claim — `C_thermal 6e-7` with exactly that
unit string in `default.param`. **[HIGH]**

### SPEC-044 — Conduction-driven evaporation of shell gas into the bubble
**Claim.** Classical (unsaturated) conductive evaporation of a cold cloud/shell of radius `R`
embedded in gas at temperature `T_h`:

```
    Ṁ_evap = 16 π μ C T_h^{5/2} R / ( 25 k_B )
```

**Source.** (b) Literature (Cowie & McKee 1977, the classical-evaporation result; the same physics
Weaver applies at the bubble wall).
**Regime.** Unsaturated: the mean free path of conducting electrons ≪ `R`; saturation parameter
`σ₀ ≲ 1`. For `σ₀ ≳ 1` the flux saturates and this formula over-predicts.
**Consistency with `default.param`.** The `cooling_boost_kappa` docstring reports a *measured*
`Ṁ ∝ C^{2/7}` (matching El-Badry+19 "Eq. 47"), with `1.2175` measured vs `2^{2/7} = 1.2190`
analytic at `f_κ = 2`. Note the **apparent tension**: the Cowie–McKee formula above is linear in
`C`, not `C^{2/7}`. Both can be true because `T_h` itself depends on `C` through SPEC-042
(`T_b ∝ C^{−2/7}`), giving `Ṁ ∝ C · (C^{−2/7})^{5/2} = C^{1−5/7} = C^{2/7}` ✓.
**This is a derived (c) reconciliation and a good regression test.** **[HIGH]**

### SPEC-045 — `LOW CONFIDENCE`: Weaver's numerical prefactors and equation numbers
**What the audit asked for, and what I can honestly deliver.**
The task asked for "the exact coefficients — e.g. the constants in Weaver+77 Eq. 20 and Eq. 37 —
and say which equation number each really is." **I could not open Weaver+77** (§0.3). I therefore
refuse to assert equation numbers. What I *can* state:

1. **The radius law coefficient is verified independently** (SPEC-050): `(250/308π)^{1/5} = 0.76293`,
   confirmed by a third-party source found via search *and* re-derived here from scratch.
2. **The interior-profile exponents are verified**: `T ∝ (1−x)^{2/5}`, `n ∝ (1−x)^{−2/5}`,
   `T_b ∝ L^{8/35} n₀^{2/35} t^{−6/35}` — the last confirmed by search *and* re-derived from
   SPEC-042.
3. **The interior-profile numerical prefactors are NOT verified and are mutually inconsistent in
   the literature I could reach.** The commonly-quoted forms are

   ```
       T_b = 1.51×10⁶ K · L₃₆^{8/35} n₀^{2/35} t₆^{−6/35}       (widely cited)
       T_b = 2.07×10⁶ K · L₃₆^{8/35} n₀^{2/35} t₆^{−6/35}       (also found in the literature)
       n_b = 4.02×10⁻³ cm⁻³ · L₃₆^{6/35} n₀^{19/35} t₆^{−22/35} (widely cited)
   ```

   **Internal-consistency test I ran (c).** The bubble must be isobaric: `P_b = n_tot k_B T`. Using
   the dynamical `P_b` I derive in SPEC-052 (`P_b/k_B = 2.5×10⁴ K cm⁻³` at `L₃₆ = n₀ = t₆ = 1`,
   `μ = 1`), and the quoted prefactor product `n_b T_b = 4.02×10⁻³ × 1.51×10⁶ = 6.07×10³ K cm⁻³`,
   the two disagree by a factor **≈4.2** — far more than the `n_tot/n_H = 2.3` composition factor
   could explain. With the `2.07×10⁶` prefactor the mismatch is ≈3.0. **At most one of these
   prefactor pairs can be right, and I cannot tell which.**
   **Audit consequence:** if the code hard-codes any of these prefactors, that is a place to look.
   Prefer the *structural* forms (SPEC-024, SPEC-042) which are prefactor-free.
   **[LOW]** for the prefactors, **[HIGH]** for the inconsistency finding.

---

## 4. Known analytic limits (these are the executable tests)

All limits below assume: constant mechanical luminosity `L_w` (or constant momentum input `ṗ`),
uniform or power-law ambient medium, no gravity, no radiation, no external pressure, shell starting
from `R = 0` at `t = 0`. These are the configurations the code must reproduce when those terms are
switched off.

### SPEC-050 — Energy-driven bubble, uniform medium (the primary validation test)
**Claim.**

```
    R2(t) = ξ_E ( L_w / ρ₀ )^{1/5} t^{3/5} ,     ξ_E = (250 / 308π)^{1/5} = 0.762934…
    v2(t) = (3/5) R2/t = 0.457760 ( L_w/ρ₀ )^{1/5} t^{−2/5}
```

**Derivation (c), in full**, because this is the load-bearing test:
Thin shell `M = (4/3)πρ₀R³`; momentum `d(MṘ)/dt = 4πR²P`; energy `dE/dt = L_w − 4πR²ṘP` with
`E = 2πPR³`. Put `R = A t^{3/5}`. Momentum ⇒ `P = (ρ₀/3)(R R̈ + 3Ṙ²) = 0.28 ρ₀ A² t^{−4/5}`.
Then `E = 2πPR³ = 1.75929 ρ₀A⁵t` and `4πR²ṘP = 2.11115 ρ₀A⁵`, so
`L_w = (1.75929 + 2.11115) ρ₀A⁵ = 3.87045 ρ₀A⁵`, giving
`A⁵ = L_w/(3.87045 ρ₀) = 0.258364 L_w/ρ₀` and `0.258364 = 250/(308π)` ✓.
**Independent confirmation (b).** A third-party source located by search states verbatim: *"the
coefficient ξ was found to be (250/308π)^(1/5) = 0.76"*.
**Numeric anchors (c), computed here:**

| convention | `R2` at `L₃₆ = 1`, `n_H = 1 cm⁻³`, `t = 1 Myr` | `v2` |
|---|---|---|
| `ρ₀ = n_H m_H` (μ = 1, Weaver's own) | **28.0 pc** | 16.4 km s⁻¹ |
| `ρ₀ = 1.4 n_H m_H` (TRINITY's `mu_convert`) | **26.2 pc** | 15.4 km s⁻¹ |

**AUDIT TRAP.** The famous "28 pc" number is a `μ = 1` number. TRINITY declares
`mu_convert = 1.4` (mass per H nucleus). A validation test that asserts 28 pc against a TRINITY run
with `n_H = 1 cm⁻³` will be **7% wrong in radius and 30% wrong in swept mass** — and would falsely
"pass" a code that had a compensating `μ` bug. Always state the `μ` convention with the number.
**[HIGH]**

### SPEC-051 — Energy budget partition in the Weaver limit
**Claim.** For a radiative outer shock (thin shell), the injected energy partitions as

```
    E_bubble  = (5/11)  L_w t = 0.45455 L_w t = (35/77) L_w t
    E_kin,sh  = (15/77) L_w t = 0.19481 L_w t
    radiated  = (27/77) L_w t = 0.35065 L_w t
```

**Derivation (c).** From SPEC-050: `E_b = 1.75929 ρ₀A⁵t` and `L_w t = 3.87045 ρ₀A⁵ t`, ratio
`= 5/11` exactly. `E_kin = ½MṘ² = (2π/3)(0.36)ρ₀A⁵t = 0.75398 ρ₀A⁵t`, ratio `15/77`.
**Audit test.** In a gravity-free, radiation-free, non-radiative-bubble run, `Eb/(L_mech·t)` must
approach `0.4545` in the energy phase. This is a *dimensionless* test — immune to unit-conversion
bugs, which makes it the ideal first check. **[HIGH]**

### SPEC-052 — Bubble pressure in the Weaver limit
**Claim.**

```
    P_b = 5 L_w t / ( 22 π R2³ )  =  0.162979 · L_w^{2/5} ρ₀^{3/5} t^{−4/5}
```

**Numeric (c):** at `L₃₆ = 1`, `n_H = 1 cm⁻³`, `t = 1 Myr`:
`P_b/k_B = 2.5×10⁴ K cm⁻³` (μ = 1) or `3.1×10⁴ K cm⁻³` (μ_H = 1.4).
**Derivation (c).** `P_b = E_b/(2πR2³)` with `E_b = (5/11)L_w t`. **[HIGH]**

### SPEC-053 — Energy-driven bubble, power-law medium (general `w`)
**Claim.** For `ρ(r) = ρ_ref (r/r_ref)^{−w}` with `w = |α| ∈ [0, 2]`:

```
    R2 ∝ t^{η} ,     η = 3/(5 − w)
    E_bubble / (L_w t) = 1/(1 + 2η)
```

and the full prefactor is

```
    A^{5−w} = L_w (3−w) / [ 4π ρ_ref r_ref^{w} · η · ((4−w)η − 1) · (½ + η) ]
```

**Derivation (c).** `M = 4πρ_ref r_ref^w R^{3−w}/(3−w) ≡ B R^{3−w}`; substituting `R = A t^η` into
the momentum + energy pair gives `L_w = B A^{5−w} η [(4−w)η − 1](½ + η)` and
`E_b = (B/2)A^{5−w}η[(4−w)η−1]t`; their ratio is `1/(1+2η)`. Checks: `w = 0 ⇒ η = 3/5`,
`E_b/(L_w t) = 5/11` ✓ (SPEC-051); `w = 2 ⇒ η = 1`, `E_b = E_kin = radiated = 1/3` each.
**Cross-check (a).** `paper_radiusComparison.py` uses `exp_weaver = 3.0/(5.0 − abs(alpha_rho))` ✓
— the published figure's "pure energy (wind)" reference line uses exactly this exponent.
**[HIGH]**

### SPEC-054 — Momentum-driven limit
**Claim.** For constant momentum injection `ṗ` into a uniform medium, starting from rest:

```
    R2(t) = ( 3 ṗ / (2π ρ₀) )^{1/4} t^{1/2} ,      v2 = R2/(2t)
```

and in a power law, `R2 ∝ t^{2/(4−w)}`.
**Derivation (c).** `M v = ṗ t` ⇒ `(4/3)πρ₀R³Ṙ = ṗ t` ⇒ `(π/3)ρ₀R⁴ = ṗ t²/2` ⇒ result.
**Cross-check (a).** `paper_radiusComparison.py` uses `exp_mom = 2.0/(4.0 − abs(alpha_rho))` ✓
(`= 1/2` for `α = 0`).
**AUDIT TRAP.** `M v = ṗ t` presumes the shell starts from rest *and* that all injected momentum is
retained. If the code instead solves `d(Mv)/dt = ṗ` from a non-zero `R₀, v₀`, the `t^{1/2}` law is
only asymptotic. **[HIGH]**

### SPEC-055 — Photoionized (D-type) limit
**Claim.** The classical Spitzer D-type expansion of an ionization-bounded HII region:

```
    R_HII(t) = R_St ( 1 + (7/4) c_i t / R_St )^{4/7}    →   R ∝ t^{4/7}
    R_St = ( 3 Q_i / (4π α_B χ_e n_H²) )^{1/3}
```

Hosokawa & Inutsuka (2006) give the variant `R = R_St(1 + (7/4)√(4/3) c_i t/R_St)^{4/7}`, which is
the same power law with a `√(4/3) = 1.1547` faster clock.
**Symbols.** `c_i` = isothermal sound speed of the ionized gas
`= sqrt(k_B T_ion/(μ_ion,shell m_H))`; at `T = 10⁴ K` and `μ = 14/22 = 0.636`,
`c_i = 11.7 km s⁻¹` (c, computed: `sqrt(1.380649e-16 × 1e4/(0.636×1.6737e-24)) = 1.166e6 cm/s`).
**Cross-check (a).** `paper_radiusComparison.py` uses `exp_spitzer = 4.0/(7.0 − 2.0*abs(alpha_rho))`
✓ (`= 4/7` for uniform).
**`AMBIGUOUS`.** Spitzer's `7/4` and Hosokawa–Inutsuka's `(7/4)√(4/3)` differ by 15% in the clock;
the published TRINITY figure anchors its reference lines to the simulation's own energy-phase
midpoint, which cancels the prefactor entirely — so the figure tests only the **exponent**, not the
coefficient. The audit should not read agreement in that figure as validating any prefactor.
**[HIGH]** for the exponent, **[MED]** for the coefficient.

### SPEC-056 — Free-expansion / early-time limit
**Claim.** Before appreciable mass is swept (`M_sw ≪ M_ejected`), the wind free-streams and
`R ∝ t` at `~v_w`. The energy-driven similarity solution is an **attractor**, so the run's initial
condition only needs to be inside its basin; but a code that starts at `R₀` with the *wrong* `v₀`
will show a transient that decays as `t^{−2/5}` relative to the similarity branch.
**Audit test.** In a clean energy-driven run, `α = v2 t/R2` should relax to `0.6` and stay there for
the duration of the energy phase. `α` is already a first-class quantity (`cool_alpha`). Plotting
the *measured* `α(t)` against 0.6 is a direct, cheap validation. **[HIGH]**

### SPEC-057 — What the published `radiusComparison` figure does and does not prove
**Claim.** The figure plots TRINITY's `R2(t)` against three power laws, each **anchored to the
TRINITY curve at the midpoint of the energy phase** (`energy_phase_midpoint`) — that is,
normalisation is imposed, not tested. The comparison therefore validates **slopes only**. Also, the
"WARPFIELD" curve in that figure is *not* WARPFIELD: it is TRINITY with `include_PHII = False`
(folders `_yesPHII`/`_noPHII`).
**Source.** (a) TRINITY-claim — `paper_radiusComparison.py` docstring and
`compute_anchored_power_law`. **[HIGH]** — record this so no audit claim over-reads that figure.

---

## 5. Density profiles

### SPEC-060 — Power-law sphere: definition
**Claim.**

```
    n(r) = n_core                      ,  r ≤ r_core
         = n_core (r/r_core)^{α}       ,  r_core < r ≤ r_cloud
         = n_ISM                       ,  r > r_cloud
    ρ(r) = μ_H m_H n(r) ,   μ_H = mu_convert = 1.4  (mass per hydrogen nucleus)
```

with `−2 ≤ α ≤ 0`; `α = 0` homogeneous, `α = −2` singular-isothermal-like.
**Source.** (a) TRINITY-claim — `docs/source/parameters.rst` gives this piecewise form verbatim.
**[HIGH]**

### SPEC-061 — Power-law sphere: enclosed mass
**Claim.**

```
    M(<r) = (4π/3) ρ_core r³                                              , r ≤ r_core
    M(<r) = 4π ρ_core [ r_core³/3 + ( r^{3+α} − r_core^{3+α} ) / ((3+α) r_core^{α}) ] , r_core < r ≤ r_cloud
    M(<r) = M_cloud                                                       , r > r_cloud
```

**Derivation (c).** Direct integration of `4πr²ρ(r)`; valid for `α ≠ −3` (always true here).
**Cross-check (a).** Identical expression appears in `paper/methods/figures/paper_densityProfile.py`
(`M_arr[reg2] = 4π ρ_core (rCore³/3 + (r^{3+α} − rCore^{3+α})/((3+α) rCore^α))`). **[HIGH]**

### SPEC-062 — Cloud radius
**Claim.** `r_cloud` is the root of `M(<r_cloud) = M_cloud` (see SPEC-005 for *which* cloud mass).
Homogeneous closed form: `r_cloud = (3 M_cloud/(4π ρ_core))^{1/3}`.
**Validation gate.** `rCloud_max` (default 200 pc) rejects implausibly diffuse clouds; separate
checks require `n_edge ≥ n_ISM` and mass consistency.
**Source.** (c) Derived; (a) TRINITY-claim (`rCloud_max`, `validate_gmc` behaviour described in
`docs/source/running.rst` and `param/paperII_grid_sweep.param` comments). **[HIGH]**

### SPEC-063 — "Core radius" and "cloud radius" — definitions
**Claim.** `r_core` is the **inner flattening radius** of the power law, *not* a physical core mass
scale; the profile is flat inside it. `r_cloud` is the **outer edge** where the cloud density meets
the ambient ISM. `r_core` is ignored for `α = 0`.
**AUDIT TRAP.** `default.param` ships `rCore = 0.01 pc` — three orders of magnitude below the
project's own guidance in `CLAUDE.md` (*"tests and scratch configs use physically plausible values
… e.g. rCore ≈ 1 pc"*), and `param/cloud_example_PL.param` uses `rCore 5`. With `α = −2` and
`r_core = 0.01 pc`, `n(1 pc) = n_core × 10⁻⁴`, so the default `rCore` makes the `α ≠ 0` profile
extraordinarily centrally concentrated. Since the **default** `densPL_alpha = 0`, `rCore` is inert
by default — but any sweep that varies `α` without also setting `rCore` inherits `0.01 pc`.
**[HIGH]**

### SPEC-064 — Bonnor–Ebert sphere: defining ODE
**Claim.** An isothermal, self-gravitating sphere in hydrostatic equilibrium obeys the isothermal
Lane–Emden equation:

```
    (1/ξ²) d/dξ ( ξ² dψ/dξ ) = e^{−ψ} ,     ψ(0)=0 , dψ/dξ|₀ = 0
    ρ(ξ) = ρ_c e^{−ψ(ξ)} ,   r = ξ · r₀ ,   r₀ = c_s / sqrt(4πGρ_c) = sqrt( k_B T /( μ m_H 4π G ρ_c) )
```

**Enclosed (dimensionless) mass:** `m(ξ) ≡ ξ² dψ/dξ`, so
`M(<r) = 4π ρ_c r₀³ · ξ² dψ/dξ`.
**Source.** (b) Literature (Ebert 1955; Bonnor 1956); (c) Derived for the mass integral (integrate
the ODE once).
**Cross-check (a).** `paper_densityProfile.py` calls `solve_lane_emden()` and uses `le_sol.f_m(ξ)`
and `le_sol.f_rho_rhoc(ξ)`, with `M(<r) = M_cloud · m(ξ)/m(ξ_out)` — i.e. the dimensionless mass
function normalised at the outer radius. ✓ **[HIGH]**

### SPEC-065 — Bonnor–Ebert critical parameters
**Claim.**

| Quantity | Critical value |
|---|---|
| dimensionless outer radius `ξ_crit` | **6.451** |
| centre-to-edge density contrast `Ω = ρ_c/ρ_edge` | **14.04** |
| dimensionless mass `m_crit` in `M = m c_s⁴/(G^{3/2}P_ext^{1/2})` | **1.18** |

Spheres with `Ω > 14.04` are gravitationally unstable.
**Source.** (b) Literature — all three confirmed by search: *"critical dimensionless radius
(ξ_crit ≈ 6.45)"*, *"a critically stable Bonnor-Ebert sphere is characterized through an
overpressure (density contrast) of 14.04"*, *"the critical radius, pressure and mass are 0.41, 1.40
and 1.18 respectively in dimensionless units"*.
**Cross-check (a).** `docs/source/parameters.rst`: *"Values above the critical Ω ≈ 14.04 are
gravitationally unstable"*; `default.param` ships `densBE_Omega = 14.1` — **marginally
supercritical, i.e. formally unstable**. That is presumably deliberate (a collapsing cloud), but it
means the default BE cloud is *not* an equilibrium configuration and its "hydrostatic" density
profile is being used outside its own validity regime. **Flag for the audit.** **[HIGH]**

### SPEC-066 — `AMBIGUOUS`: which temperature sets the BE scale radius
`r₀ = c_s/sqrt(4πGρ_c)` requires an isothermal sound speed, hence a temperature and a mean
molecular weight. TRINITY has three candidates in `default.param`: `TShell_neu = 100 K` with
`mu_mol = 14/6` (molecular — the physically right choice for a GMC), `TShell_ion = 10⁴ K`, or an
implicit one. The `.param` schema exposes **no** cloud temperature parameter at all; instead the BE
sphere is parameterised by `(M_cloud, n_core, Ω)`, from which `r₀` and hence the implied `c_s` is
*back-solved*. That is self-consistent, but it means **the BE cloud's implied temperature is an
output, not an input, and may be unphysical.**
**Audit test.** For each shipped BE example (`param/cloud_example_BE.param`: `M = 10⁶ M⊙`,
`n_core = 10⁴ cm⁻³`, `Ω = 14.1`), back out `c_s` and hence `T` from `r₀` and check it is ~10–30 K.
If it lands at 10³ K, the profile is a mathematical fit, not a physical cloud.
**Source.** (c) Derived; (a) for the parameter list. **[HIGH]** that this is unspecified.

---

## 6. Feedback input (SPS)

### SPEC-070 — What an SB99 table must supply
**Claim.** The required canonical columns (loader will not start without them) are `t`, `Qi`,
`Lbol`, `Lmech_W`, `pdot_W`, plus either `fi` or both `Li` and `Ln`, plus either `Lmech_total` or
`Lmech_SN`. Optional/derivable: `Lmech_total`, `Lmech_SN`, `pdot_SN`, `Mdot_SN`, `v_SN`, `Li`, `Ln`.
**Physical meanings and conventional cgs units:**

| Canonical | Physical quantity | cgs | TRINITY internal (AU) |
|---|---|---|---|
| `t` | cluster age | s | Myr |
| `Qi` | H-ionizing photon rate (>13.6 eV) | s⁻¹ | Myr⁻¹ |
| `fi` | ionizing fraction of `Lbol` | – | – |
| `Lbol` | bolometric luminosity | erg s⁻¹ | M⊙ pc² Myr⁻³ |
| `Li`, `Ln` | ionizing / non-ionizing luminosity | erg s⁻¹ | M⊙ pc² Myr⁻³ |
| `Lmech_W` | wind mechanical luminosity | erg s⁻¹ | M⊙ pc² Myr⁻³ |
| `pdot_W` | wind momentum injection rate | g cm s⁻² (dyn) | M⊙ pc Myr⁻² |
| `Lmech_SN`, `Lmech_total` | SN / total mechanical luminosity | erg s⁻¹ | M⊙ pc² Myr⁻³ |
| `pdot_SN` | SN momentum rate | dyn | M⊙ pc Myr⁻² |
| `Mdot_SN` | SN mass-loss rate | g s⁻¹ | M⊙ Myr⁻¹ |
| `v_SN` | SN ejecta velocity | cm s⁻¹ | pc Myr⁻¹ |

**Source.** (a) TRINITY-claim — `docs/source/parameters.rst` §*Custom SPS files*, which lists
exactly these canonicals, `cgs` aliases, and AU targets. (b) Literature: Starburst99 does output
mechanical luminosity, wind momentum flux, and ionizing photon rates for H I / He I / He II.
**[HIGH]**

### SPEC-071 — Wind terminal velocity and mass-loss rate from `(L, ṗ)`
**Claim.** For a wind of mass-loss rate `Ṁ_w` and terminal speed `v_w`:

```
    L_w = ½ Ṁ_w v_w²  ,   ṗ_w = Ṁ_w v_w
    ⇒   v_w = 2 L_w / ṗ_w  ,   Ṁ_w = ṗ_w² /( 2 L_w )
```

**Derivation (c).** Trivially from the definitions.
**Unit sanity (c).** In AU, `(M⊙ pc² Myr⁻³)/(M⊙ pc Myr⁻²) = pc Myr⁻¹` ✓.
**Order-of-magnitude anchor [LOW].** For a `10⁶ M⊙` cluster at `t < 3 Myr`, SB99 gives
`L_w ~ 10⁴⁰ erg s⁻¹` and `v_w ~ 1500–2500 km s⁻¹`, so `ṗ_w ~ 10³² dyn`. Compare
`L_bol/c ~ 3×10³² dyn` for `L_bol ~ 10⁴³ erg s⁻¹`: **direct radiation pressure exceeds the wind
momentum by ~3× at early times** — a well-known result that any correct force-budget plot must
show. **[HIGH]** for the relations, **[LOW]** for the anchors.

### SPEC-072 — Cold-gas mass loading
**Claim.** `FB_mColdWindFrac` and `FB_mColdSNFrac` add entrained cold mass to `Ṁ_w`/`Ṁ_SN`. At
fixed `L`, adding mass must **reduce the effective velocity** (`v_eff = sqrt(2L/Ṁ_total)`) and
**increase the momentum** (`ṗ = sqrt(2 L Ṁ_total)`). Both default to 0.
**Source.** (a) TRINITY-claim (`default.param`: "increases Mdot_wind, reduces velocity"); (c) for
the scalings. **[HIGH]**

### SPEC-073 — Linear mass scaling of the SPS table
**Claim.** `f_mass = M_cluster / sps_refmass` multiplies **every mass-scaled canonical** (all except
`t`, `fi`, `v_SN`) after unit conversion. `sps_refmass = 10⁶ M⊙` for the bundled table.
**Source.** (a) TRINITY-claim — `docs/source/parameters.rst`: *"Mass-scaled canonicals (everything
except `t`, `fi`, `v_SN`) are multiplied by `f_mass` after unit conversion."*
**VALIDITY LIMIT (b/c) — an audit finding waiting to happen.** Linear scaling of an SPS table is
valid only when the IMF is **fully sampled**, i.e. `M_cluster ≳ 10⁴–10⁵ M⊙`. Below that,
stochastic sampling of the massive-star IMF makes `Q_i`, `L_w`, and `ṗ_w` scatter by orders of
magnitude and their *means* deviate systematically. `param/paperII_grid_sweep.param` sweeps
`mCloud` down to `10⁴ M⊙` with `sfe` down to `0.01` — that is `M_cluster = 100 M⊙`, i.e. an
expected massive-star count below one. **Those grid cells are outside the model's validity, and
nothing in the schema flags it.** **[HIGH]**

### SPEC-074 — Ionizing split
**Claim.** `Q_i` counts photons above 13.6 eV; `f_i = L_i/L_bol` is the ionizing luminosity
fraction. `docs/source/parameters.rst` says supplying `Li` and `Ln` explicitly *"bypasses the
hardcoded 13.6 eV ionizing-fraction split"* — i.e. there **is** a hard-coded split when only `fi`
is available. The audit should locate it and confirm consistency: `L_i + L_n = L_bol` must hold.
**Source.** (a) TRINITY-claim. **[HIGH]**

---

## 7. Cooling

### SPEC-080 — Two-regime cooling assembly
**Claim.** The net cooling rate is assembled from two tables split at `T = 10^{5.5} K ≈ 3.16×10⁵ K`:

```
    T > 10^{5.5} K :  CIE curve   Λ_CIE(T, Z)
    T < 10^{5.5} K :  non-CIE (photoionization-aware) CLOUDY/OPIATE cube
```

**Source.** (a) TRINITY-claim — `default.param`: `path_cooling_CIE` *"Selects the CIE (T > 10^5.5 K)
cooling curve"*; `path_cooling_nonCIE` *"Folder containing the non-CIE (T < 10^5.5 K) OPIATE/CLOUDY
cubes"*.
**Physical justification (c).** Above ~3×10⁵ K, collisional ionization dominates and the local
radiation field is irrelevant to the ionization state, so CIE is a good approximation. Below it, the
gas is photoionized by the cluster and the ionization state (hence the cooling) depends on the local
radiation field — a CIE curve would be badly wrong there. **[HIGH]**

### SPEC-081 — CIE table: content and dimensions
**Claim.** A CIE cooling curve tabulates the cooling **efficiency** `Λ(T)` in `erg cm³ s⁻¹`, such
that the volumetric cooling rate is a *product of two densities* times `Λ`. Bundled options:
`1 → coolingCIE_1_Cloudy.dat`, `2 → coolingCIE_2_Cloudy_grains.dat`,
`3 → coolingCIE_3_Gnat-Ferland2012.dat` (the default).
**Source.** (a) TRINITY-claim for the file list; (b) Literature — Gnat & Ferland 2012, ApJS 199, 20,
*"Ion-by-ion cooling efficiencies"*, computed with Cloudy v10.00 for `10⁴ ≤ T ≤ 10⁸ K`, **low
density, optically thin**.
**Validity regime (b).** Optically thin, low density (no density-dependent level populations),
collisional ionization equilibrium, no photoionization, no dust cooling (except in table 2). CIE
assumes ionization equilibrium — invalid for rapidly cooling gas, which is exactly the bubble
interior in the transition phase. **[HIGH]** for scope, **[MED]** for the specific file mapping.

### SPEC-082 — `AMBIGUOUS` and error-prone: the cooling normalisation
**The three conventions in circulation:**

```
    (i)   dE/dV/dt = n_e n_H Λ(T)      ← most common for "cooling efficiency" tables
    (ii)  dE/dV/dt = n_e n_ion Λ(T)    ← Gnat & Ferland tabulate ion-by-ion this way
    (iii) dE/dV/dt = n_tot² Λ(T)  or  n_H² Λ(T)
```

For fully ionized H+He at `x_He = 0.1`, `Z_He = 2`: `n_e = 1.2 n_H`, `n_tot = 2.3 n_H`. Converting
(i)→(iii, with n_H²) is a factor 1.2; (i)→(iii, with n_tot²) is a factor `2.3²/1.2 = 4.4`.
**I could not confirm which convention the Gnat & Ferland file ships in.** The audit must read the
table header and check the code's multiplier.
**Why this matters more than usual here.** `L_cool` enters the transition trigger
(`(L_gain−L_loss)/L_gain ≤ 0.05`, SPEC-013) *directly*. A factor-of-few normalisation error moves
the energy→momentum transition time, which is the code's headline prediction.
**Source.** (c) Derived; (b) partially confirmed. **[HIGH]** that this is a real ambiguity.

### SPEC-083 — Non-CIE table: content and dimensions
**Claim.** The OPIATE/CLOUDY cubes tabulate the **net** cooling *minus* heating rate for
photoionized gas as a function of at least `(n, T, ionization parameter)`, where the ionization
parameter is typically `U = Φ/(n c)` or the photon flux `Φ = Q_i/(4πr²)`. Per-age files are selected
at runtime from `SB99_rotation` + `ZCloud`.
**Source.** (a) TRINITY-claim — `docs/source/parameters.rst`; the description "cubes" implies ≥3
axes, and *"Per-age files are selected at runtime"* implies a 4th (cluster age) axis handled by file
selection rather than interpolation.
**AUDIT TRAP (c).** Selecting a *file* per age rather than interpolating in age means the cooling
function is **piecewise-constant in cluster age** — a source of discontinuities in the ODE RHS. Look
for jumps in `L_cool` at the age-table boundaries. **[MED]**

### SPEC-084 — Net rate assembly
**Claim.** The correct assembly is

```
    (du/dt)_rad = − Λ_net(n, T, Φ, Z) · (density product per SPEC-082)
```

where `Λ_net` is cooling minus photoheating (which can be **negative**, i.e. net heating, in the
photoionized regime). A code that treats the non-CIE table as pure cooling will over-cool the
`~10⁴ K` gas and, in particular, will fail to hold the photoionized shell layer at `T_ion = 10⁴ K`.
**Source.** (c) Derived. **[MED]** — I cannot confirm the table's sign convention.

### SPEC-085 — Metallicity restriction
**Claim.** `ZCloud` is declared to support only `1.0` (solar), and `SB99_rotation = 0` is rejected
with the bundled tables. Any run with `Z ≠ 1` is outside the shipped tables' coverage.
**Source.** (a) TRINITY-claim — `docs/source/parameters.rst`: *"**Currently only solar (1) is
supported.**"*
**Contradiction to flag.** `default.param` nevertheless defines `dust_noZ = 0.05 Z⊙` ("metallicity
below which there is effectively no dust") and `dust_sigma` is described as "at solar metallicity",
both of which only make sense if `Z` varies. And `docs/source/running.rst` documents sweeping
`ZCloud = [0.5, 1.0]` as a worked example of the folder-naming rules. **The schema, the docs, and
the naming examples disagree about whether `Z ≠ 1` is legal.** **[HIGH]**

---

## 8. Unit conventions

### SPEC-090 — The unit system
**Claim.** Parameter-file inputs are **cgs extended by M⊙ and Myr** (plus pc for length, cm⁻³ for
number density, km/s for velocity, K for temperature). Internally TRINITY works in
**`[M⊙, pc, Myr]`** ("AU" = astronomical units in this codebase's sense). Conversion is driven by
the `# UNIT:` annotations in `default.param`.
**Source.** (a) TRINITY-claim — `docs/source/parameters.rst` §*Unit system*. **[HIGH]**

### SPEC-091 — Conversion table (computed here)
**Claim.** With `pc = 3.0856775814913673×10¹⁸ cm`, `Myr = 3.15576×10¹³ s` (Julian year),
`M⊙ = 1.98892×10³³ g`:

| Quantity | AU unit | cgs value | note |
|---|---|---|---|
| length | pc | `3.0857e18 cm` | |
| time | Myr | `3.15576e13 s` | Julian yr `3.15576e7 s` |
| mass | M⊙ | `1.98892e33 g` | (IAU nominal `1.98841e33`; 0.03% spread) |
| velocity | pc/Myr | `9.77781e4 cm/s` = **0.977781 km/s** | `1 km/s = 1.022712 pc/Myr` |
| energy | M⊙ pc² Myr⁻² | **`1.90148e43 erg`** | |
| luminosity | M⊙ pc² Myr⁻³ | **`6.0255e29 erg/s`** | |
| force / `ṗ` | M⊙ pc Myr⁻² | **`6.1623e24 dyn`** | |
| pressure | M⊙ pc⁻¹ Myr⁻² | **`6.4721e-13 dyn cm⁻²`** | |
| mass density | M⊙ pc⁻³ | **`6.7696e-23 g cm⁻³`** | |
| number density | pc⁻³ | `1 cm⁻³ = 2.9380e55 pc⁻³` | |
| `G` | pc³ M⊙⁻¹ Myr⁻² | **`4.4985e-3`** | `= 4.30091e-3 pc M⊙⁻¹ (km/s)²` |
| `k_B` | M⊙ pc² Myr⁻² K⁻¹ | **`7.261e-60`** | `= 1.380649e-16 erg/K` |

**Source.** (c) Derived — every entry computed in this document from the three base conversions.
**[HIGH]** (arithmetic; re-derivable).

### SPEC-092 — Classically error-prone conversions (rank-ordered for the audit)
**Claim.** In order of how often they bite:

1. **`n_H → ρ`.** `ρ = μ_H m_H n_H` with `μ_H = mu_convert = 1.4` — *mass per hydrogen nucleus*,
   **ionisation-independent**. Using `μ_ion = 14/23 ≈ 0.609` (mass per *particle*) instead inflates
   `n` by `2.3×` or deflates `ρ` by the same. `paper_densityProfile.py` carries an explicit warning
   comment about exactly this. **The `μ` used for `ρ ↔ n` and the `μ` used for `P = ρ k T/(μ m_H)`
   are different constants and must not be interchanged.**
2. **`P = n_tot k_B T` vs `n_H k_B T`.** `n_tot/n_H = 2.3` (hot bubble), `2.2` (ionized shell),
   `1.1` (neutral atomic), `0.6` (molecular). All four are declared consistently in `default.param`
   via `mu_ion`, `mu_ion_shell`, `mu_atom`, `mu_mol`; the risk is picking the wrong one for a
   region.
3. **Fifth powers.** `R ∝ (L/ρ)^{1/5}` means a 1% error in a length conversion is a 5% error in the
   inferred `L`. The Weaver coefficient must be evaluated in one consistent system.
4. **`P/k` vs `P`.** `PISM` is declared in `K cm⁻³` (i.e. `P/k_B`), not pressure. Multiply by
   `k_B = 1.380649e-16` to get `dyn cm⁻²`. `param/paperII_grid_sweep.param` sweeps `PISM` up to
   `10⁶ K cm⁻³` = `1.4×10⁻¹⁰ dyn cm⁻²`, which is a *large* confining pressure.
5. **`Myr` vs `10⁶ yr` vs `t₆`.** Weaver's `t₆` is `t/10⁶ yr` — the same as Myr, but only if the
   year is the same year. Julian (`3.15576e7 s`) vs tropical (`3.1557e7`) differ by `2×10⁻⁵` —
   negligible; `3.15e7` (a common shortcut) is off by 0.2%, which is 1% in `L` via point 3.
6. **`km/s` vs `pc/Myr`.** They differ by only 2.3%, which makes the error *hard to see* and
   therefore especially dangerous. `paper_teaser.py` documents `v2` is stored in `pc/Myr` and must
   be multiplied by `cvt.v_au2kms`.
7. **Dust opacity units.** `κ_IR` is `cm² g⁻¹` (per gram of gas) but `σ_d` is `cm²` (per hydrogen
   nucleus). Mixing them is a `~10²³` error, which at least fails loudly.

**Source.** (c) Derived + (a) corroborating comments in `paper/` and `default.param`. **[HIGH]**

---

## 9. Stopping conditions

### SPEC-100 — The physically meaningful end states
**Claim.** A 1-D feedback-bubble run must end in exactly one of:

| Fate | Physical meaning | Correct trigger |
|---|---|---|
| **Dispersal / dissolution** | The shell has been diluted to ambient; the cloud is gone | shell peak density falls below `n_ISM` and stays there |
| **Re-collapse** | Feedback lost; gravity wins; the shell falls back | `v2 < 0` sustained, `R2 → small` |
| **Escape / blowout** | The shell leaves the cloud and never returns | `R2 > r_cloud` **and** `v2 > v_esc(R2)` |
| **Stall** | Shell halts at finite radius in pressure equilibrium | `v2 → 0` with `dv2/dt ≈ 0` |
| **Feedback exhausted** | Cluster stops supplying energy/momentum | end of the SPS table / cluster lifetime |
| **Numerical cutoff** | Not physics — a run limit | `t > stop_t`, `R2 > stop_r` |

**Source.** (c) Derived; (a) TRINITY-claim for the ones the schema exposes. **[HIGH]**

### SPEC-101 — What TRINITY actually exposes
**Claim.**

| Parameter | Default | Trigger |
|---|---|---|
| `allowShellDissolution` | True | enables the dissolution check |
| `stop_t_diss` | 1 Myr | duration `shell_nMax` must stay below `nISM` before dissolution fires |
| `stop_r` | 500 pc | max shell radius (`None` disables) |
| `stop_t` | 15 Myr | max simulated duration (`None` disables) |
| `stop_at_rCloud_nSnap` | None | terminate `N` snapshots after `R2 > r_cloud` |
| `coll_r` | 1 pc | "radius below which the cloud is considered completely collapsed" |

**Source.** (a) TRINITY-claim — `default.param`, `docs/source/parameters.rst`. **[HIGH]**

### SPEC-102 — `AMBIGUOUS`: dissolution criterion
**Reading A.** `shell_nMax < n_ISM` for `stop_t_diss` is a *density* criterion: the shell can no
longer be distinguished from the ambient medium. Defensible.
**Reading B.** The physically standard dissolution criterion is that the shell becomes
**subsonic / pressure-confined** relative to the ambient medium (`v2 < c_s,ISM`), or that it
fragments (gravitational/Rayleigh–Taylor instability) — neither of which is a density threshold.
**Concern (c).** With the default `n_ISM = 1 cm⁻³` and a *swept* shell, `shell_nMax` should exceed
`n_ISM` by the shock compression ratio (≥4 for a strong adiabatic shock, ≫4 for an isothermal one)
essentially always, *unless* the shell is thickening faster than it sweeps. The criterion therefore
fires only in a fairly extreme regime; the audit should check how often it actually triggers versus
`stop_t`/`stop_r`. **[MED]**

### SPEC-103 — `AMBIGUOUS`: `coll_r` is not a collapse criterion
**Claim.** `coll_r = 1 pc` ("radius below which the cloud is considered completely collapsed") is a
**radius threshold**, not a dynamical criterion. Physically, re-collapse should be declared when
`v2 < 0` *and* the shell is gravitationally bound (`½v2² < G(M_cluster+M_sh)/R2`), independent of
any absolute radius. A fixed 1 pc threshold is scale-dependent: for a `10⁹ M⊙` cloud with
`r_cloud ~ 100 pc` (the `paperII` sweep reaches `mCloud = 5e9`), a shell that has fallen from 100 pc
to 2 pc has manifestly collapsed but would not trip a 1 pc test.
**Source.** (a) TRINITY-claim for the parameter; (c) Derived for the objection. **[HIGH]**

### SPEC-104 — `AMBIGUOUS`: `R2 > r_cloud` is not escape
**Claim.** Crossing the cloud edge means the shell has entered the ambient ISM; it does **not** mean
the shell escapes. A shell can cross `r_cloud` with `v2 < v_esc` and still turn around. Conversely,
`stop_at_rCloud_nSnap` terminates the run *at* the crossing, which forecloses the question. Note the
default is `None` (do not stop), but every shipped sweep example sets it explicitly.
**Source.** (c) Derived; (a) for the parameter semantics. **[HIGH]**

### SPEC-105 — Termination bookkeeping
**Claim.** `metadata.json` must record `termination = {exit_code, outcome, detail, timestamp,
model_name}` plus a `final_state` block and a `termination_debug` block containing a last-two-snapshot
diff, a NaN/Inf inventory, and physics sanity checks.
**Source.** (a) TRINITY-claim — `docs/source/running.rst`, *metadata.json*.
**Audit use.** `termination_debug`'s NaN/Inf inventory is a ready-made corpus for a survey of which
quantities go non-finite and when. **[HIGH]**

---

## 10. Index of ambiguities and unverifiable claims

Ranked by expected audit value.

| # | SPEC | Issue | Why it matters |
|---|---|---|---|
| 1 | SPEC-030, SPEC-023 | **`P_HII` geometry and the `max()` combination rule** are unspecified in every source I may read, yet they *are* TRINITY's headline contribution over WARPFIELD. `max()` is non-differentiable and breaks the shell↔bubble work balance (SPEC-035). | The central physics claim rests on an under-documented choice. |
| 2 | SPEC-045 | **Weaver+77 equation numbers and interior prefactors unverifiable** (arXiv/ADS blocked). Worse, the two commonly-quoted prefactor pairs are **mutually inconsistent with isobaricity by a factor ≈3–4** when checked against the dynamical `P_b`. | Any hard-coded `1.51e6`/`4.02e-3` in the code cannot be validated from this spec; prefer the structural forms SPEC-024/042. |
| 3 | SPEC-014, SPEC-015 | **The energy→momentum transition has no unique criterion**; TRINITY's default is a 5% numerical threshold, and the published Paper-II grid runs with `cooling_boost_fmix = 4` — i.e. **not** default physics. | The headline output (transition time, dispersal vs re-collapse) is threshold- and knob-dependent. |
| 4 | SPEC-005, SPEC-066 | **Cloud normalisation and BE temperature are back-solved, not declared.** Post-SFE vs total mass changes `r_cloud` by 11% at `ε = 0.3`; the BE sphere's implied `c_s`/`T` is an unchecked output, and the default `Ω = 14.1` is formally unstable. | Silent 10–30% errors in swept mass; a BE "equilibrium" that is not one. |
| 5 | SPEC-082, SPEC-092 | **Cooling normalisation (`n_e n_H` vs `n_e n_ion` vs `n²`) and the `μ` used for `n ↔ ρ`** are the two classic factor-of-2-to-4 traps, and `L_cool` feeds the transition trigger directly. | A factor of 2 here moves the phase transition, hence the science result. |
| 6 | SPEC-020, SPEC-035 | **Ram-pressure double-counting** in the EOM, and **work-balance mismatch** between `P_drive` (shell side) and `P_b dV/dt` (bubble side). | Both are silent energy non-conservation. |
| 7 | SPEC-073, SPEC-085 | **Validity-regime violations shipped in the repo**: `M_cluster = 100 M⊙` grid cells (IMF not sampled); `Z ≠ 1` swept in the docs while the schema says solar-only. | The published grid contains cells the model cannot represent. |
| 8 | SPEC-025, SPEC-036, SPEC-055 | O(1) convention choices: the `3/4` strong-shock factor in `R1`; enthalpy (`5/2 P`) vs internal energy (`3/2 P`) in the vent flux; Spitzer vs Hosokawa–Inutsuka D-type clock. | Each is a 15–70% effect on a subsidiary quantity. |
| 9 | SPEC-063, SPEC-103, SPEC-104 | Defaults that are physically odd (`rCore = 0.01 pc`) or scale-dependent (`coll_r = 1 pc`, `R2 > r_cloud` as "escape"). | Cheap to check, easy to get wrong in a sweep. |
| 10 | SPEC-083 | Age-indexed cooling **files** rather than age interpolation ⇒ piecewise-constant `L_cool(t_cluster)`. | Discontinuous ODE RHS; solver chatter. |

---

## 11. Suggested executable tests, in dependency order

Cheapest and most diagnostic first. All are derived from the SPEC claims above and are
**dimensionless or convention-free wherever possible**, so they cannot be faked by a compensating
unit bug.

| # | Test | Passes iff | SPEC |
|---|---|---|---|
| T1 | Ionizing-photon budget closure | `f_gas + f_dust + f_esc = 1` at every snapshot, to machine precision | SPEC-028 |
| T2 | Force-budget closure | recorded forces reproduce `M_sh dv2/dt` to integrator tolerance | SPEC-007 |
| T3 | Weaver energy fraction | `E_b/(L_mech t) → 5/11 = 0.4545` in a gravity-/radiation-free energy-phase run | SPEC-051 |
| T4 | Weaver `α` | measured `α = v2 t/R2 → 0.6` through the energy phase | SPEC-041, 056 |
| T5 | Conduction closure | `δ ≈ (2/7)(2α − β − 1)` for the solved `(α,β,δ)` in the energy phase | SPEC-042 |
| T6 | Weaver radius, `μ`-explicit | `R2 = 0.762934 (L_w/ρ₀)^{1/5} t^{3/5}`, with `ρ₀ = 1.4 n_H m_H` ⇒ 26.2 pc at `L₃₆=n_H=t₆=1`, **not** 28 pc | SPEC-050 |
| T7 | Power-law slopes | `R2 ∝ t^{3/(5−w)}` for `w = 0, 1, 2` in the pure-energy limit | SPEC-053 |
| T8 | Momentum limit | `R2 ∝ t^{1/2}` with coefficient `(3ṗ/(2πρ₀))^{1/4}` once `P_b → 0` | SPEC-054 |
| T9 | Isobaricity of the interior | `n(r)·T(r)` constant across the stored `bubble_n_arr`/`bubble_T_arr` profiles | SPEC-040 |
| T10 | `ξ = 0.98` factor | reported `T0` relates to the profile's central `T_b` by `(1−0.98)^{2/5} = 0.209` | SPEC-040 |
| T11 | Composition consistency | `n_tot/n_H = 2.3` (bubble), `2.2` (ionized shell), `1.1` (atomic), `0.6` (molecular) wherever `P = n k T` is formed | SPEC-092 |
| T12 | Bubble/shell work balance | `∮ P_drive dV` (shell side) equals `∮ P_b dV` (bubble side) unless documented otherwise | SPEC-035 |
| T13 | Phase-boundary continuity | `dv2/dt` has no jump > tolerance across `energy→implicit→transition→momentum` | SPEC-016 |
| T14 | `C_f = 1` identity | `coverFraction = 1.0` reproduces the sealed bubble **bit-identically** | SPEC-036 |
| T15 | BE implied temperature | back-solved `c_s` from `r₀` for `param/cloud_example_BE.param` lands at `T ~ 10–30 K` | SPEC-066 |
| T16 | Enclosed-mass identity | `M(<r_cloud)` from the profile integral equals the normalising cloud mass to `<1e-10` relative | SPEC-061, 062 |

---

## 12. Change log

- **2026-07-29** — Created. Built from `README.md`, `docs/source/*.rst`, `paper/`, `param/*.param`,
  `trinity/_input/default.param`, and external literature reachable only via `WebSearch` (all
  non-GitHub HTTPS blocked by the egress proxy; see §0.3). No implementation file read, no code run.
