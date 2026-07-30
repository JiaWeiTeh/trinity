# S8 shell structure — Lens C (what it should be)

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

**Method.** Derived from first principles + standard ISM radiative transfer, from the interface
alone (`get_shellODE(y, r, f_cover, is_ionised, params)`, `shell_structure_pure(params) ->
ShellProperties`, plus the constant *names* `_NSHELL_MAX`, `_SHELL_ODE_MXSTEP`). No implementation,
comment, or docstring was read. Spec cross-references are to
`docs/dev/code-audit/reference/PHYSICS_SPEC.md` (SPEC-nnn), the only permitted `docs/dev/` read.

**Confidence discipline.** Every numeric constant below is tagged `[recalled]` (from memory, may be
off) or `[computed]` (arithmetic done here, re-derivable). Literature access was blocked, so no
citation was verified.

---

## 0. What the interface forces the physics to be

`get_shellODE(y, r, f_cover, is_ionised, params)` has the shape of a **right-hand side for an
initial-value problem in radius**:

- `r` is the independent variable ⇒ the integration variable is radius, not mass or optical depth.
- `is_ionised` is a **separate argument, not a component of `y`** ⇒ the ionisation state is *not*
  evolved as a continuous fraction. It is a piecewise switch. That is the **sharp-front (x = 1 inside,
  x = 0 outside) approximation**, and it carries a specific, checkable validity condition (§2.4).
- `f_cover` enters the RHS ⇒ the covering fraction modifies either the local column or the local
  photon flux (§3.4). It must enter **exactly once**.
- `_SHELL_ODE_MXSTEP` is the classic `scipy.integrate.odeint(..., mxstep=)` knob; `_NSHELL_MAX` caps
  the radial sample count. Both are **failure-mode surfaces** (§6).

---

## 1. The shell structure ODE system

### 1.1 Geometry and the physical setup

The swept-up shell occupies `R2 ≤ r ≤ R_out ≡ R2 + ΔR`, where `R2` is the contact discontinuity with
the hot bubble (SPEC-002 zone 3). It is irradiated **from inside** by the cluster at `r = 0`
(`Q_i` ionising photons s⁻¹, `L_i` + `L_n = L_bol`). Interior to the ionisation front the gas is at
`T_ion ≈ 10⁴ K`; exterior it is neutral at `T_neu ~ 10–10² K`.

### 1.2 Momentum (quasi-hydrostatic) equation

In the frame co-moving with the shell, the steady momentum equation for a fluid element is

```
    dP/dr  =  −ρ(r) [ a_sh + G M(<r)/r² ]  +  f_rad(r)                          (S8.1)
```

with, term by term:

| term | sign | dimension (cgs) | meaning |
|---|---|---|---|
| `dP/dr` | — | dyn cm⁻³ | thermal pressure gradient |
| `−ρ a_sh` | **negative** for outward acceleration `a_sh = dv2/dt > 0` | dyn cm⁻³ | inertial pseudo-force in the accelerating shell frame; points **inward** |
| `−ρ G M(<r)/r²` | negative | dyn cm⁻³ | self/cluster gravity; negligible across `ΔR ≪ R2` |
| `+f_rad` | **positive** (outward) | dyn cm⁻³ | momentum deposited by absorbed photons |

**Sign check.** With radiation only (`a_sh = 0`), `dP/dr = +f_rad > 0`: pressure and density
**increase outward** through the shell, because radiation entering at the inner face pushes matter
against the outer (ram-confined) edge. With inertia only, `dP/dr < 0`. Integrating (S8.1) across the
whole shell reproduces the thin-shell equation of motion (SPEC-020) exactly:

```
    4πR2² [ P(R2) − P(R_out) ]  +  F_rad  =  M_sh a_sh  +  F_grav              (S8.2)
```

**This is the single strongest cross-tier invariant in S8:** the shell-structure ODE and the global
EOM are *the same equation*. If the structure ODE drops `−ρ a_sh`, its `P(R_out)` is no longer the
physical outer pressure and must not be fed back into the dynamics as one.

### 1.3 The radiation force density — the exact form

The force per unit volume is `(1/c) ×` (energy **absorbed** per unit volume per unit time). Splitting
by absorber and band:

```
    f_rad(r) = (1/c) [ ⟨hν_i⟩ ( n_HI σ_HI + n_H σ_d ) Φ_i(r)  +  σ_d n_H F_n(r) ]   (S8.3)
```

- `Φ_i(r)` [cm⁻² s⁻¹] — ionising **photon number** flux;
- `F_n(r) = L_n e^{−τ_d(r)} / (4π r²)` [erg cm⁻² s⁻¹] — attenuated non-ionising energy flux;
- `⟨hν_i⟩` — mean ionising photon energy, ~15–18 eV for a young cluster `[recalled, medium]`.

A defensible lumped form is `f_rad = (n_H σ_d / c) · L_bol e^{−τ}/(4πr²)` (dust dominant), which is
the standard shell-structure simplification.

**Trap (sign/structure).** The force is `κ ρ F / c ≡ (dτ/dr)·F/c`, **not** `−d(F/c)/dr`. The latter
includes the geometric dilution `2F/(rc)`, which transfers **no** momentum to the gas. In a thin
shell the error is O(ΔR/R2) ~ 10⁻⁴ (harmless); in a thick shell it is O(1) and has the wrong sign
structure.

### 1.4 The full system (isothermal within each layer)

Define `ψ ≡ n_tot/n_H`: `ψ_ion = 2.2`, `ψ_atom = 1.1`, `ψ_mol = 0.6` (SPEC-092 item 2), so
`P = ψ n_H k_B T`. The minimal well-posed state vector is **3–4 components**:

**Ionised branch** (`is_ionised = True`, `T = T_ion`):

```
 (a)  dn_H/dr = f_rad(r) / (ψ_ion k_B T_ion)                 [cm⁻³ cm⁻¹]
 (b)  dQ/dr   = − 4π r² [ α_B χ_e n_H²  +  σ_d n_H Φ_i ] ,   Φ_i = Q/(4π r²)   [s⁻¹ cm⁻¹]
 (c)  dτ/dr   = σ_d n_H                                       [cm⁻¹]
 (d)  dM/dr   = 4π r² μ_H m_H n_H                             [g cm⁻¹]
```

**Neutral branch** (`is_ionised = False`, `T = T_neu`, `Q ≡ 0`):

```
 (a')  dn_H/dr = σ_d n_H F_n(r) / (c ψ_neu k_B T_neu)
 (b')  dQ/dr   = 0            (Q is already zero)
 (c')  dτ/dr   = σ_d n_H
 (d')  dM/dr   = 4π r² μ_H m_H n_H
```

**Why `Q` and not `Φ`.** Written in `Φ`, (b) carries a geometric term
`dΦ/dr = −(2/r)Φ − α_B χ_e n² − σ_d n Φ`. That term is not a physical sink; carrying it in the state
variable makes photon conservation a *numerical* result rather than an *exact* one. Integrating `Q`
makes photon number conservation manifest and is the numerically safer choice. **Either is
acceptable; mixing them (using `Φ` in the sink terms but forgetting `4πr²` when accumulating
absorbed photons) is a `4πr²` error.**

### 1.5 Well-posedness: direction, boundary data, free boundary

Integration must run **outward**, `r: R2 → R_out`:

- The radiation field is a **pure downstream problem** under the on-the-spot (OTS) approximation —
  no diffuse field is transported back inward — so the transfer equations are an IVP from the inner
  face. Integrating inward would require shooting on the transmitted flux (ill-conditioned: the
  transmitted flux is exponentially small).
- Inner boundary data (3 conditions at `r = R2`): `P(R2) = P_drive` ⇒ `n_H(R2) = P_drive/(ψ_ion k_B T_ion)`;
  `Q(R2) = f_cover-normalised Q_i`; `τ(R2) = 0`; `M(R2) = 0`.
- Outer boundary is **free**, fixed by the terminal condition `M(R_out) = M_sh` (SPEC-021).

So it is a **free-boundary IVP terminated by an event**, not a fixed-interval quadrature. A correct
implementation must root-find on the mass condition (event-based termination or bisection on `ΔR`);
integrating a pre-guessed interval and interpolating is acceptable only if the residual
`|M(R_out) − M_sh|/M_sh` is checked.

**Over-determination check.** Four conditions (`P(R2)`, mass closure, photon IC, and the physical
outer pressure `P(R_out) = P_ext + ρ_amb v2²`) for three unknowns. The resolution is (S8.2): the
inertial term absorbs the mismatch. Imposing both boundary pressures *and* dropping `−ρ a_sh` is
inconsistent.

---

## 2. The ionisation front

### 2.1 Locating condition

The front is where the **stellar** ionising budget is exhausted:

```
    Q(R_IF) = 0    ⟺    Q_i = ∫_{R2}^{R_IF} 4π r² [ α_B χ_e n_H²(r) + σ_d n_H Φ_i(r) ] dr   (S8.4)
```

This is the shell-geometry generalisation of SPEC-029, and SPEC-029's annulus form
`Q_i = (4π/3) α_B χ_e n_H² (R_i³ − R_in³)` is its constant-`n`, dust-free special case ✓.

In the plane-parallel limit `ΔR ≪ R2` (which holds overwhelmingly — see §2.5) it collapses to

```
    ΔR_ion  ≈  Q_i / ( 4π R2² α_B χ_e n_H² )                                    (S8.5)
    N_ion   ≡  n_H ΔR_ion  =  Φ_i(R2) / ( α_B χ_e n_H )                        (S8.6)
```

### 2.2 Local ionisation–recombination balance

```
    (1 − x) n_H σ_HI Φ_i  =  α_B n_e n_p  =  α_B χ_e x² n_H²                    (S8.7)
```

with `n_e = χ_e x n_H`, `n_p = x n_H`, `χ_e = 1 + x_He = 1.1` for singly-ionised He (SPEC-029). **The
`n²` in the recombination term is `n_H²`, not `n_tot²` and not `(ρ/m_H)²`** — see §7.

`α_B`, **case B**, is the correct coefficient because the OTS approximation assumes the ionised layer
is optically thick to its own recombination Lyman continuum. `α_B(10⁴ K) = 2.59×10⁻¹³ cm³ s⁻¹`
`[recalled, high]`; `α_A(10⁴ K) ≈ 4.18×10⁻¹³ cm³ s⁻¹` `[recalled, medium]`; ratio **1.61**.
`α_B ∝ T^{−0.7…−0.8}` over 5×10³–2×10⁴ K `[recalled, medium]`.

### 2.3 Photon budget consumption

Photons leave the budget by exactly two channels inside the shell, plus escape:

```
    Q_i  =  Q_gas  +  Q_dust  +  Q_esc
    Q_gas  = ∫ 4πr² α_B χ_e n_H² dr ,   Q_dust = ∫ 4πr² σ_d n_H Φ_i dr ,  Q_esc = Q(R_out)
    ⇒  f_gas + f_dust + f_esc = 1     to machine precision                      (SPEC-028, T1)
```

### 2.4 Exhausted before the outer edge (ionisation-bounded) — the required behaviour

`Q → 0` at `R_IF < R_out`. Then:

1. `f_esc = 0` **exactly** (not "small").
2. The equation set must switch: drop recombination and LyC pressure, set `T = T_neu`, `ψ = ψ_neu`.
3. **The density must jump *up* discontinuously.** The front in a quasi-static shell is a
   **weak D-type** front: mass flux through it is subsonic, so the momentum jump condition reduces
   to pressure continuity `ψ_ion n_ion T_ion = ψ_neu n_neu T_neu`, hence

```
    n_neu / n_ion  =  (ψ_ion T_ion) / (ψ_neu T_neu)                             (S8.8)
      = (2.2 × 10⁴)/(1.1 × 10²)  =  200          (atomic, T_neu = 100 K)   [computed]
      = (2.2 × 10⁴)/(0.6 × 10 )  =  3667         (molecular, T_neu = 10 K) [computed]
```

   Carrying `n` **continuously** across the switch while changing only `T`/`ψ` silently imposes a
   *pressure*-discontinuous (R-type) front, which drops the shell's internal pressure by 200–3700×
   at the front. That is the single most consequential structural choice in S8.
4. Because `dn/dr → ∞` at the front, a single continuous ODE call cannot represent it. The
   integration must be **split into two segments** with an explicit junction — which is exactly what
   an `is_ionised` *argument* (rather than state component) enables. The caller must **root-find**
   `Q = 0`; locating the front by grid resolution alone under-resolves it (§2.5).

### 2.5 Not exhausted (density-bounded) — the required behaviour

`Q(R_out) > 0`. The shell is **fully ionised through**, there is no neutral layer, `f_esc =
Q(R_out)/Q_i > 0`, and `T = T_ion` everywhere. Downstream consequences a correct implementation must
honour:

- `f_esc` is a headline science output (SPEC-001: "how much ionising radiation escapes") — it must
  come from this branch and be exactly `Q(R_out)/Q_i`, not a fitted or clipped value.
- A fully-ionised shell **cannot be compressed below the pressure of its own 10⁴ K gas**. This is the
  physical origin of `P_HII` as a *floor* on the shell's inner pressure and the only non-circular
  reading of `max(P_b, P_HII)` (SPEC-022/030). If instead `P_HII` is computed as
  `ψ_ion n_H(R2) k_B T_ion` with `n_H(R2)` itself back-solved from `P_drive`, then
  `P_HII ≡ P_drive` identically and `max()` is a no-op — a circularity the audit should test for
  directly (does `P_HII` ever exceed `P_b` in a run?).

### 2.6 Validity of the sharp-front (boolean `is_ionised`) approximation

From (S8.5) and the front thickness `ℓ_IF ~ 1/(n_H σ_HI)`:

```
    ℓ_IF / ΔR_ion  =  4π R2² α_B χ_e n_H / ( σ_HI Q_i )                         (S8.9)
```

`[computed]` For `Q_i = 10⁴⁹ s⁻¹`, `R2 = 5 pc`, `σ_HI = 6.3×10⁻¹⁸ cm²` `[recalled, high]`:
`n_H = 10⁴ ⇒ 0.14`; `n_H = 10⁵ ⇒ 1.4`. **The sharp-front approximation therefore breaks down at
`n_H ≳ 10⁵ cm⁻³` — a density the compressed shell of a `nCore = 10⁵ cm⁻³` cloud (the TRINITY
default, SPEC-003) will reach.** In that regime `x < 1` throughout, recombinations run at `α_B χ x²
n²` (slower), and the true ionised thickness is larger than (S8.5) by ~`1/x²`. Assuming `x = 1`
therefore **over-consumes the photon budget and under-predicts `ΔR_ion` and `f_esc`** — a *biased*,
not random, error. Regime finding, not a bug per se, but it must be flagged rather than silent.

---

## 3. Optical depth and radiation pressure

### 3.1 The three optical depths (they are different numbers)

| symbol | definition | dimension | `τ = 1` at |
|---|---|---|---|
| `τ_LyC,dust` | `σ_d N_H` over the **ionised** layer only | – | `N_H = 6.7×10²⁰ cm⁻²` `[computed]` |
| `τ_UV` | `σ_d N_H` over the **whole** shell | – | same |
| `τ_IR` | `κ_IR Σ_sh = κ_IR M_sh/(4πR2²)` | – | `Σ = 0.25 g cm⁻²` ⇒ `N_H = 1.07×10²³ cm⁻²` `[computed, κ_IR=4]` |

Consistency anchor `[computed]`: `σ_d = 1.5×10⁻²¹ cm²/H` ⇒ per-gram UV opacity
`σ_d/(μ_H m_H) = 1.5e−21/(1.4×1.6735e−24) = 640 cm² g⁻¹`, i.e. `κ_UV/κ_IR ≈ 160`. **A shell becomes
UV-thick 160× earlier than it becomes IR-thick.** For `M_sh = 10⁵ M⊙` at `R2 = 10 pc`:
`Σ = 0.0166 g cm⁻²`, `τ_UV = 10.6`, `τ_IR = 0.066` `[computed]` — the typical regime is
**UV-thick / IR-thin**, so the IR term should be a few-percent correction, not a dominant one.

### 3.2 Direct radiation pressure

```
    F_rad,dir = (L_bol/c) f_abs ,   f_abs = 1 − e^{−τ_UV}                       (SPEC-026)
```

`τ_UV` **must be the one the shell ODE integrated** (state component `τ`), not an independently
parameterised column. If the ODE integrates the local force (S8.3), then
`∫ f_rad 4πr² dr → (L/c)(1−e^{−τ})` automatically in the plane-parallel limit — so the internal
force and the net force are the *same* physics viewed twice, not two forces. Adding an internal
radiation-force term to the shell ODE **and** an independent `F_rad,dir` to the EOM is fine (the
former sets the profile, the latter the net momentum); adding `F_rad,dir` *twice* to the EOM is not.

### 3.3 Reprocessed IR — the form that does **not** double-count

Only the *absorbed* luminosity can be reprocessed. Correct:

```
    F_rad,IR = τ_IR · L_abs / c  =  τ_IR (L_bol/c)(1 − e^{−τ_UV})
    ⇒  F_rad,total = (L_bol/c) (1 − e^{−τ_UV})(1 + τ_IR)                        (S8.10)
```

SPEC-027's quoted additive form `(L/c)(1 − e^{−τ_UV} + τ_IR)` **fails the optically-thin limit**: as
`τ_UV → 0` it leaves a residual `τ_IR L/c` of reprocessed radiation from luminosity that was never
absorbed. With `κ_UV/κ_IR ≈ 160` the two forms agree to <1% in the UV-thick regime, so this is a
correctness-of-form issue that only bites in thin/dissolving shells — but it is the regime where the
dissolution criterion (SPEC-102) fires. **Which forms double-count if both applied, plainly:**

1. `(L/c)(1−e^{−τ_UV})` **plus** `(L/c)τ_IR` computed on the *full* `L` — double-counts the
   reprocessing of un-absorbed light (small in the thick limit, unbounded in the thin limit).
2. LyC momentum counted **both** through `⟨hν_i⟩ Q_i/c` **and** inside `L_bol/c` — `L_bol` already
   contains `L_i`. Bookkeeping must be either (`L_i` via `Q_i⟨hν⟩`, `L_n` via `L_n`) or (`L_bol`
   with a single `τ`), never both. Magnitude: `f_i` is typically 0.1–0.3 of `L_bol` for a young
   cluster `[recalled, medium]`, so this is a 10–30% force error.
3. `P_HII` used as the driving pressure at `R2` **and** the ionised layer counted as part of the
   shell mass with its own independent inner pressure — the same gas pushes twice.
4. `f_cover` applied to both the photon budget and the column (§3.4).

### 3.4 The `f_cover` normalisation — the one physically consistent convention

If a fraction `(1 − f_cover)` of the solid angle is open:

- Photons are emitted isotropically, so the **flux per unit covered area is unchanged**:
  `Φ_i(R2) = Q_i/(4πR2²)`.
- The shell mass is spread over solid angle `4π f_cover`, so the **column is enhanced**:
  `Σ_patch = M_sh/(4π R2² f_cover)`, i.e. the mass-accumulation ODE (d) carries `f_cover`.
- Escape bookkeeping: `f_esc,total = (1 − f_cover) + f_cover · f_esc,patch`.
- Net radiation force on the shell: `f_cover × (L/c)(1 − e^{−τ_patch})`.

Scaling `Φ_i(R2)` by `f_cover` *and* enhancing the column by `1/f_cover` applies the correction
twice. Note this `f_cover` is dimensionally and physically distinct from the venting `C_f` of
SPEC-036; if they are the same parameter, `C_f = 1` must reproduce the sealed case bit-identically
(SPEC test T14).

### 3.5 Trapping condition

IR trapping is active when `τ_IR > 1` ⇔ `Σ_sh > 1/κ_IR = 0.25 g cm⁻²` ⇔
`M_sh/(4πR2²) > 0.25`, i.e. `M_sh > 3.1×10⁵ M⊙ (R2/10 pc)²` `[computed]`. Below that the boost
must vanish smoothly. Above `τ_IR ~ few` the single-scattering×`τ` estimate over-predicts (SPEC-027
validity note): real 3-D shells leak through low-column channels.

---

## 4. Dimensions and the unit-conversion mandate

| quantity | symbol | cgs | AU (`M⊙, pc, Myr`) |
|---|---|---|---|
| radius | `r`, `R2`, `ΔR` | cm | pc |
| H-nuclei density | `n_H` | cm⁻³ | pc⁻³ (`1 cm⁻³ = 2.938×10⁵⁵ pc⁻³`) |
| mass density | `ρ = μ_H m_H n_H`, `μ_H = 1.4` | g cm⁻³ | M⊙ pc⁻³ |
| pressure | `P = ψ n_H k_B T` | dyn cm⁻² | M⊙ pc⁻¹ Myr⁻² |
| force density | `f_rad` | dyn cm⁻³ | M⊙ pc⁻² Myr⁻² |
| photon rate | `Q` | s⁻¹ | Myr⁻¹ |
| photon flux | `Φ_i` | cm⁻² s⁻¹ | pc⁻² Myr⁻¹ |
| energy flux | `F_n` | erg cm⁻² s⁻¹ | M⊙ Myr⁻³ |
| recombination coeff. | `α_B` | cm³ s⁻¹ | pc³ Myr⁻¹ |
| photoion. cross-sec. | `σ_HI` | cm² **per H atom** | pc² |
| dust cross-sec. | `σ_d` | cm² **per H nucleus** | pc² |
| dust opacity | `κ_IR` | cm² **g⁻¹** | pc² M⊙⁻¹ |
| column | `N_H` | cm⁻² | pc⁻² |
| mass column | `Σ` | g cm⁻² | M⊙ pc⁻² |
| optical depth, `x`, `f_*` | — | dimensionless | dimensionless |

**Where conversion is mandatory (this module is the worst offender in the code).** The shell ODE is
the one place that multiplies *micro-physics constants that are only ever quoted in cgs*
(`α_B`, `σ_d`, `σ_HI`, `k_B`, `c`, `m_H`) by *dynamical quantities the code carries in AU*
(`r`, `n`, `Q`, `L`, `P`). Every product must be brought into one system before the RHS is formed.
Conversions `[computed]`, using `pc = 3.0857×10¹⁸ cm`, `Myr = 3.15576×10¹³ s`:

- `α_B = 2.59×10⁻¹³ cm³ s⁻¹  =  2.78×10⁻⁵⁵ pc³ Myr⁻¹`
- `σ_d = 1.5×10⁻²¹ cm²  =  1.58×10⁻⁵⁸ pc²`
- `σ_d n_H Δr` is dimensionless **only** if all three share a length unit: at `n_H = 10³ cm⁻³`,
  `Δr = 1 pc`, `τ_d = 1.5e−21 × 1e3 × 3.0857e18 = 4.63` — dropping the pc→cm factor gives
  `τ = 1.5×10⁻¹⁸`, i.e. a perfectly transparent shell and `f_esc → 1`. **This failure is silent
  and plausible-looking, not loud.**
- `σ_d` (per H nucleus) vs `κ_IR` (per gram) differ by `μ_H m_H = 2.34×10⁻²⁴`; swapping them is a
  `4×10²³` error (loud), but multiplying `σ_d` by `ρ` instead of `n_H` is a `2.34×10⁻²⁴` error
  (silent, gives `τ ≈ 0`).
- `ρ ↔ n_H` uses `μ_H = 1.4`; `P = ρ k T/(μ m_H)` uses `μ_ion,shell = 14/22`. **Different constants**
  (SPEC-092 item 1). Using `μ = 14/22` for `ρ ↔ n` inflates `n` by 2.2× ⇒ recombination rate by
  4.84×.

Numeric anchors for the reconciler `[computed]`:
`c_s,ion = sqrt(k_B T_ion/(μ_ion,shell m_H)) = 1.14×10⁶ cm s⁻¹ = 11.4 km s⁻¹` (≈ SPEC-055's 11.7,
difference is `m_H` convention);
`R_St(Q=10⁴⁹, n=10³, χ=1.1) = 2.03×10¹⁸ cm = 0.66 pc`;
`ΔR_ion(Q=10⁴⁹, R2=5 pc, n=10⁴) = 1.17×10¹⁴ cm = 3.8×10⁻⁵ pc` ⇒ `ΔR/R2 = 7.6×10⁻⁶`.

---

## 5. Exact invariants the implementation must satisfy

| # | Invariant | Tolerance |
|---|---|---|
| **I1** | `∫_{R2}^{R_out} 4πr² ρ dr = M_sh` (all swept gas is in the shell, SPEC-021) | integrator tol; residual must be **checked**, not assumed |
| **I2** | `f_gas + f_dust + f_esc = 1` at every snapshot (SPEC-028) | machine precision |
| **I3** | `Q(r) ≥ 0` and **monotonically non-increasing** outward | exact (sign guard) |
| **I4** | `n_H(r) > 0` and finite everywhere | exact |
| **I5** | `dP/dr ≥ 0` within a layer when `f_rad ≥ 0` and inertia is neglected; `dP/dr` sign must match the sign convention chosen in (S8.1) | exact |
| **I6** | `x ∈ [0,1]`; with the boolean switch, `x ≡ 1` inside / `0` outside and nothing between | exact |
| **I7** | `τ ≥ 0`, monotonically non-decreasing outward | exact |
| **I8** | `ΔR/R2 ≪ 1` in the thin-shell regime; **if `ΔR ≳ R2` the thin-shell EOM (SPEC-020) and the `4πR2²` area factors are invalid** and the run must flag it | `ΔR/R2 < 0.1` is a reasonable alarm |
| **I9** | Pressure continuity across the ionisation front ⇒ density jump (S8.8) | exact ratio |
| **I10** | Structure/EOM consistency: `4πR2²[P(R2) − P(R_out)] + F_rad − F_grav = M_sh a_sh` (S8.2 ≡ SPEC-020, SPEC-007 T2) | integrator tol |
| **I11** | Dust-free, radiation-free, uniform-`n` limit must reduce (S8.4) **exactly** to the annulus Strömgren balance `Q_i = (4π/3)α_B χ_e n²(R_IF³ − R2³)` (SPEC-029) | machine precision — this is a *free* unit test |
| **I12** | `shell_nMax` (feeds the dissolution stop, SPEC-101/102) must be the max over the *converged* profile, and must exceed `n_ISM` by ≥ the shock compression ratio while the shell is a real shock | consistency |
| **I13** | `n_H(R2)` used for `P_HII` must be the same `n_H(R2)` the ODE started from (no second, independent density) | exact |

---

## 6. Behaviour on integration failure

**What "failure" means here.** (i) `_SHELL_ODE_MXSTEP` exceeded; (ii) `_NSHELL_MAX` radial points
exhausted before `M(r) = M_sh`; (iii) non-finite `n`, `Q`, or `τ`; (iv) `n → 0` or negative; (v) the
outward integration reaching `r` far beyond any plausible `R_out` without closing the mass.

**Which downstream quantities become undefined.** *All* of: `ΔR`, `R_out`, `n(r)`, `shell_nMax`,
`τ_UV` ⇒ `f_abs` ⇒ `F_rad,dir`, `τ_IR` ⇒ `F_rad,IR`, `f_gas`/`f_dust`/`f_esc`, `R_IF`, the ionised
mass, and `P_HII`. Those in turn feed **`P_drive`** (SPEC-022), the **force budget** (SPEC-007),
the **published stacked-area photon-budget figure** (SPEC-028), and the **dissolution stopping
criterion** (SPEC-101). A shell-integration failure is therefore not a local numerical nuisance; it
corrupts the ODE right-hand side of the *global* dynamical system.

**Why consuming a partial profile is unsafe — and specifically, biased.** A truncated outward
integration always stops *short* of `R_out`. Therefore:

- `τ` is a **lower bound** ⇒ `f_abs = 1 − e^{−τ}` **under**-estimated ⇒ `F_rad` too small ⇒ the
  shell decelerates spuriously;
- `Q(r_last) > Q(R_out)` ⇒ **`f_esc` over-estimated** — and `f_esc` is a headline result of the paper;
- `shell_nMax` under-estimated (the density peaks at the *outer* edge, §1.2/I9) ⇒ can spuriously
  trip the dissolution criterion `shell_nMax < n_ISM`;
- `M(r_last) < M_sh` ⇒ mass is silently lost relative to SPEC-021.

Every one of these errors has a **fixed sign**. That is the crux: a partial profile does not add
noise, it adds a systematic bias in a consistent direction, so it will not average out over a sweep
and will not look like a numerical artefact in an ensemble plot.

**The specific `odeint` hazard.** `scipy.integrate.odeint` does **not raise** when `mxstep` is
exceeded: it prints `"Excess work done on this call"` to stdout and **returns the partial
trajectory**. A constant named `_SHELL_ODE_MXSTEP` therefore only protects the code if the caller
passes `full_output=True` and inspects `infodict['message']`/`ier`, or the code switches to
`solve_ivp` and checks `sol.status`/`sol.success`. Otherwise the failure mode is *exactly* the
silent-partial-profile case above, plus a stdout message no one reads in a sweep.

**What a trustworthy implementation must do.** Exactly one of:
1. **Raise**, and terminate the run with a recorded `termination.outcome`/`detail` (SPEC-105), or
2. **Fall back** to a documented closed-form approximation (e.g. the uniform-density Strömgren
   annulus, I11) *and* set a boolean validity flag on `ShellProperties` that is (a) propagated into
   the snapshot, (b) counted in the metadata, and (c) checked before `f_esc`/`f_abs` are published.

Clamping, `np.nan_to_num`, `max(x, 0)` guards, or returning the last successful row **without a
flag** converts a numerical failure into a physics result and is the failure mode this audit should
weight most heavily.

---

## 7. Known traps (each with the size of the error)

**T1 — Case A vs case B.** OTS ⇒ **case B**. `α_A/α_B = 1.61` `[recalled, medium]`. Using `α_A`
over-counts recombinations by 61% ⇒ `ΔR_ion` 1.6× too small, `R_St` 1.17× too small
(`R ∝ α^{−1/3}`), `f_esc` too small, `f_gas` too large. The correct check: the ionised layer is
optically thick to its own LyC (by construction, since it is ionisation-bounded), so case B is right
and case A would be a genuine error, not a convention.

**T2 — Uniform-density Strömgren radius used as the shell layer thickness.** `R_St = (3Q/(4πα_Bχn²))^{1/3}`
is a **filled sphere measured from the origin**. Inside a shell at `R2` the ionised volume is a thin
annulus. The relation between them `[computed]`:

```
    ΔR_ion  =  R_St³ / (3 R2²)      ⇒   ΔR_ion / R_St = (R_St/R2)² / 3
```

For `R_St = 0.14 pc` (n = 10⁴, Q = 10⁴⁹) and `R2 = 5 pc` this is `2.7×10⁻⁴` — using `R_St` as the
thickness over-predicts the ionised path length by **~3700×**. Consequences: `f_dust` grossly
over-predicted (dust absorption of LyC scales with the path length), the ionised mass over-predicted,
`P_HII` geometry wrong, and `ΔR` possibly exceeding `R2` (violating I8). SPEC-029's annulus form
`(R_i³ − R_in³)` is the correct one; the trap is any place `R_St` appears *bare*.

**T3 — `σ_d` per H used as per gram.** `σ_d = 1.5×10⁻²¹ cm²/H` vs `κ = 640 cm² g⁻¹` differ by
`μ_H m_H = 2.34×10⁻²⁴` `[computed]`. Direct swap = `4×10²³` (loud, fails). The *dangerous* variant is
`τ = σ_d · ρ · Δr` (multiplying the per-H cross-section by mass density): `τ` too small by
`2.34×10⁻²⁴` ⇒ `f_abs ≈ 0`, `f_esc ≈ 1`, zero radiation force — all finite, all plausible-looking.

**T4 — which `n` in the `n²`.** The recombination term is `α_B n_e n_p = α_B χ_e x² n_H²`. Error
factors relative to correct `[computed]`:

| wrong choice | factor on the recombination rate | factor on `ΔR_ion` |
|---|---|---|
| `n_tot² = (2.2 n_H)²` | ×4.84 | ÷4.84 |
| `(ρ/m_H)² = (1.4 n_H)²` | ×1.96 | ÷1.96 |
| `n_e² = (1.1 n_H)²` | ×1.10 | ÷1.10 |
| `n_H²` with `χ_e` omitted | ×0.909 | ×1.10 |

`R_St ∝ (α χ n²)^{−1/3}`, so these become 1.7×, 1.25×, 1.03× in radius — small enough to hide, large
enough to move `f_esc`.

**T5 — neglecting the shell's optical depth to its own recombination photons.** Using case B **is**
the statement that the diffuse LyC is reabsorbed on the spot; that is correct here. The genuinely
neglected term is **resonantly trapped Lyman-α**: ~0.68 Lyα photons per case-B recombination
`[recalled, medium]`, resonantly scattered many times, contributing an extra radiation force that can
approach the direct force at high column. Standard shell models omit it (dust destroys Lyα before it
runs away, which is why the omission is usually defensible). It should be recorded as a **known
neglected term**, not silently absent.

**T6 — grid resolution across the front.** `ΔR_ion` and `ΔR_neutral` differ by ~1–2 orders of
magnitude (density jump 200×). A **uniform** radial grid of `_NSHELL_MAX` points spanning the whole
shell cannot resolve the ionised layer (`ΔR_ion/ΔR ~ 10⁻²–10⁻³`); the front location, and hence
`f_esc` and `f_dust`, would be set by the grid spacing rather than by physics. Either the grid must
be geometric/adaptive or the two segments must be integrated separately with a root-found junction.

**T7 — `μ` and `T` switched inconsistently at the front.** The jump (S8.8) depends on `ψT`, not `T`
alone. Changing `T` from 10⁴ to 10² but leaving `ψ = 2.2` gives a jump of 100 instead of 200 — a
factor-2 error in the neutral density and hence in `shell_nMax`, `τ_UV`, and `τ_IR`. Whether the
neutral layer is atomic (`ψ = 1.1`) or molecular (`ψ = 0.6`) is another factor ~1.8.

**T8 — dust in the ionised layer.** Grains are partially destroyed/charged in H II gas; using the same
`σ_d` in both layers is the standard approximation but should scale with `Z` (SPEC-027 audit trap:
`dust_noZ = 0.05 Z⊙` implies a `Z` scaling exists — it must be applied to `σ_d` **and** `κ_IR`
consistently, or the two dust channels disagree about how much dust there is).

**T9 — geometric dilution counted as a force.** See §1.3. `f_rad = (dτ/dr)F/c`, never `−d(F/c)/dr`.

**T10 — the `P_HII` circularity.** §2.5. If `n_H(R2)` is set by `P_drive` and `P_HII` is then computed
from that same `n_H(R2)`, `max(P_b, P_HII)` is a no-op. `P_HII` must come from an independently
pinned geometry (the Strömgren balance over the available shell mass) to be a meaningful floor.

---

```json
[
  {
    "id": "S8-C-01",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "37",
    "class": "sign",
    "claim": "The shell-structure momentum equation must read dP/dr = -rho*(a_shell + GM/r^2) + f_rad, with the radiation term POSITIVE (outward) and the inertial/gravity terms NEGATIVE, so that integrating it across the shell reproduces the thin-shell EOM of SPEC-020 exactly.",
    "evidence": "In the shell co-moving frame a fluid element is in balance when -dP/dr + f_rad - rho*a = 0. Integrating 4*pi*R2^2 * dP/dr over [R2, R_out] gives 4*pi*R2^2*(P(R2)-P(R_out)) + F_rad = M_sh*a + F_grav, which is SPEC-020 term for term. Radiation entering the inner face pushes matter outward, so with inertia neglected P and n INCREASE outward and the shell's peak density sits at its outer edge.",
    "expected": "f_rad enters with + sign; inertial/gravity terms with - sign; the integrated pressure drop across the shell equals (M_sh*a - F_rad + F_grav)/(4*pi*R2^2).",
    "failure_scenario": "A sign flip on f_rad makes the shell densest at its INNER edge, inverting the location of shell_nMax, the optical-depth weighting, and the ionisation-front position; the force budget (SPEC-007 / test T2) then fails to close by 2*F_rad.",
    "repro": "Check closure of 4*pi*R2^2*(P_in - P_out) + F_rad - F_grav = M_sh*dv2/dt against the recorded snapshot forces, and check that the stored shell profile has dn/dr >= 0 within each layer.",
    "severity": "S1",
    "confidence": "high"
  },
  {
    "id": "S8-C-02",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "37",
    "class": "coefficient",
    "claim": "The radiation force density must be f_rad = (dtau/dr)*F(r)/c = (n_H*sigma_d/c)*F(r) (+ the LyC term), NOT -d(F/c)/dr.",
    "evidence": "Momentum is transferred only by absorption/scattering. The geometric dilution term 2F/(r c) contained in -d(F/c)/dr transfers no momentum to the gas. Dimensions: [F/c] = erg cm^-3 = dyn cm^-2, times [dtau/dr] = cm^-1 gives dyn cm^-3 = force per volume, correct.",
    "expected": "f_rad proportional to the local absorption coefficient times the local attenuated flux; no 2/r term in the force.",
    "failure_scenario": "Including the geometric term adds a spurious outward force of relative size 2*Delta_R/R2 (harmless, ~1e-4, in a thin shell) but becomes O(1) and wrongly signed in a thick/dissolving shell, exactly the regime where the dissolution stop fires.",
    "repro": "Set sigma_d = 0 with L_bol > 0: f_rad must be identically zero. If the code returns a nonzero dP/dr, the geometric term is being counted as a force.",
    "severity": "S3",
    "confidence": "high"
  },
  {
    "id": "S8-C-03",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "37",
    "class": "coefficient",
    "claim": "The recombination sink must be alpha_B * chi_e * x^2 * n_H^2 with n_H the HYDROGEN-NUCLEI density and chi_e = n_e/n_H = 1.1; not n_tot^2, not (rho/m_H)^2, not n_e^2.",
    "evidence": "Recombination rate per volume = alpha_B * n_e * n_p, with n_p = x*n_H and n_e = (x + x_He)*n_H = chi_e*n_H at full ionisation with singly-ionised He (x_He = 0.1). This reproduces SPEC-029's Q_i = (4pi/3)*alpha_B*chi_e*n_H^2*(R_i^3 - R_in^3) exactly.",
    "expected": "alpha_B*chi_e*n_H^2 with chi_e = 1.1, n_H in cm^-3 (or consistently converted).",
    "failure_scenario": "Using n_tot = 2.2*n_H gives 4.84x too many recombinations, shrinking the ionised layer 4.84x and the Stroemgren radius 1.7x; f_esc collapses toward zero and f_gas toward one, corrupting the published photon-budget figure. Using rho/m_H = 1.4*n_H gives 1.96x. Omitting chi_e gives 0.91x.",
    "repro": "In the dust-free, radiation-free, uniform-density limit the solved ionisation-front radius must satisfy Q_i = (4pi/3)*alpha_B*1.1*n_H^2*(R_IF^3 - R2^3) to machine precision.",
    "severity": "S1",
    "confidence": "high"
  },
  {
    "id": "S8-C-04",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "37",
    "class": "coefficient",
    "claim": "The recombination coefficient must be CASE B (alpha_B ~ 2.59e-13 cm^3 s^-1 at 1e4 K), consistent with the on-the-spot approximation that the ionised layer is optically thick to its own recombination Lyman continuum.",
    "evidence": "The shell ODE transports only the stellar ionising field (no diffuse-field transfer). That is only self-consistent under OTS, which requires case B. alpha_A(1e4 K) ~ 4.18e-13 cm^3 s^-1 [recalled], so alpha_A/alpha_B ~ 1.61.",
    "expected": "alpha_B = 2.59e-13 cm^3 s^-1 at 1e4 K (2.78e-55 pc^3/Myr), used everywhere the photon budget is consumed.",
    "failure_scenario": "Using case A over-consumes photons by 61%: ionised layer 1.6x too thin, f_esc under-predicted, f_gas over-predicted, and P_HII geometry wrong; the shell would appear ionisation-bounded in regimes where it is actually density-bounded.",
    "repro": "Compare the constant against 2.59e-13 cm^3 s^-1; check the shell f_esc against an independent Stroemgren calculation with the same alpha.",
    "severity": "S2",
    "confidence": "high"
  },
  {
    "id": "S8-C-05",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "37",
    "class": "state",
    "claim": "The ionising-photon transport equation must conserve photon number: dQ/dr = -4*pi*r^2*(alpha_B*chi_e*n_H^2 + sigma_d*n_H*Phi), with Phi = Q/(4*pi*r^2). If the state variable is the flux Phi instead, the geometric term -(2/r)*Phi must be present.",
    "evidence": "Q(r) = 4*pi*r^2*Phi(r) is the photon rate crossing radius r; only true sinks (recombination, dust) may reduce it. Written in Phi, d/dr[4 pi r^2 Phi] = 4 pi r^2 (dPhi/dr + 2 Phi / r), so the -(2/r)Phi term is mandatory and is NOT a sink.",
    "expected": "Photon closure Q_i = Q_gas + Q_dust + Q_esc, i.e. f_gas + f_dust + f_esc = 1 to machine precision (SPEC-028, test T1).",
    "failure_scenario": "Mixing the two forms (using Phi in the sink terms while accumulating absorbed photons without 4*pi*r^2, or dropping the 2/r term) breaks the budget closure by a factor of order 4*pi*r^2 or by 2*Delta_R/R2; the published stacked-area figure would no longer sum to unity.",
    "repro": "Assert f_gas + f_dust + f_esc == 1 within 1e-12 on every snapshot of param/simple_cluster.param.",
    "severity": "S1",
    "confidence": "high"
  },
  {
    "id": "S8-C-06",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": "85",
    "class": "regime",
    "claim": "At the ionisation front the density must JUMP UP by (psi_ion*T_ion)/(psi_neu*T_neu) ~ 200 (atomic, 100 K) to 3667 (molecular, 10 K); pressure, not density, is continuous.",
    "evidence": "The front is a weak D-type front in a quasi-static shell: the mass flux through it is subsonic, so the momentum jump condition P1 + rho1 v1^2 = P2 + rho2 v2^2 reduces to P1 = P2. With P = psi*n_H*k*T and psi_ion = 2.2, psi_atom = 1.1, psi_mol = 0.6 (SPEC-092), n_neu/n_ion = psi_ion*T_ion/(psi_neu*T_neu).",
    "expected": "n_H is discontinuous at R_IF by that exact ratio; P(r) is continuous; the integration is split into two segments joined at a root-found R_IF.",
    "failure_scenario": "Carrying n continuously while switching only T/psi imposes an R-type (pressure-discontinuous) front and drops the internal pressure by 200-3700x at the front. The neutral layer is then 200x too thin and 200x too rarefied: shell_nMax under-predicted (spurious dissolution trigger), tau_UV and tau_IR under-predicted (radiation force under-predicted), and the structure/EOM consistency (SPEC-020) broken.",
    "repro": "Read a stored shell profile: compute psi*n*T on both sides of the density peak/front and check continuity; check n_neu/n_ion against 200 for T_neu = 100 K.",
    "severity": "S1",
    "confidence": "high"
  },
  {
    "id": "S8-C-07",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": "85",
    "class": "silent-failure",
    "claim": "If the integration does not complete (mxstep exceeded, NSHELL_MAX exhausted, non-finite state, mass target not reached), the run must raise or set an explicit validity flag that is propagated to the snapshot; a partially-integrated profile must never be consumed silently.",
    "evidence": "A truncated OUTWARD integration always stops short of R_out, so every derived quantity is biased in a FIXED direction: tau is a lower bound so f_abs and F_rad are under-estimated; Q(r_last) > Q(R_out) so f_esc is over-estimated; shell_nMax is under-estimated (density peaks at the outer edge, S8-C-01) so the dissolution criterion can fire spuriously; M(r_last) < M_sh so SPEC-021 is violated. These are systematic biases, not noise, so they will not average out across a sweep.",
    "expected": "Either (i) raise and terminate with a recorded termination.outcome/detail (SPEC-105), or (ii) fall back to a documented closed form AND set a flag on ShellProperties that is written to the snapshot and counted in metadata.",
    "failure_scenario": "scipy.integrate.odeint does not raise on mxstep exhaustion: it prints 'Excess work done on this call' to stdout and returns the partial trajectory. In a sweep nobody reads stdout, so a numerical failure becomes a published f_esc.",
    "repro": "Force a failure (set the mxstep constant to ~10 in a copy of the harness, or run the stiffest hidens edge config); check whether the run terminates, flags, or silently continues, and whether f_esc/f_abs are still written.",
    "severity": "S1",
    "confidence": "high"
  },
  {
    "id": "S8-C-08",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": "35",
    "class": "silent-failure",
    "claim": "_SHELL_ODE_MXSTEP only protects the code if the integrator's status is inspected: odeint must be called with full_output=True and infodict/ier checked, or solve_ivp with sol.status/sol.success checked.",
    "evidence": "odeint's contract on excess work is a printed warning plus a partial return, not an exception. A bare mxstep= keyword changes when the warning fires but not whether it is detected.",
    "expected": "The return status is inspected on every call and a failure is escalated, not logged-and-continued.",
    "failure_scenario": "The constant gives the appearance of a guard while the failure remains undetected; the biased partial profile of S8-C-07 propagates into P_drive and the force budget, corrupting the global ODE right-hand side rather than just one diagnostic.",
    "repro": "Grep the call site for full_output / ier / sol.status handling; run the stiffest config and check for 'Excess work' on stdout with no corresponding termination record.",
    "severity": "S1",
    "confidence": "high"
  },
  {
    "id": "S8-C-09",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": "85",
    "class": "state",
    "claim": "The outer edge is a FREE boundary fixed by the mass closure integral(4 pi r^2 rho dr, R2..R_out) = M_sh; the residual |M(R_out) - M_sh|/M_sh must be checked, not assumed.",
    "evidence": "SPEC-021: all cloud gas interior to R2 is in the shell. The structure problem has 3 initial conditions at R2 plus one terminal condition determining R_out; it is an event-terminated IVP, not a fixed-interval quadrature.",
    "expected": "Event-based termination or bisection on Delta_R, with the mass residual asserted below a tolerance.",
    "failure_scenario": "Integrating a guessed interval and truncating gives a shell whose mass silently disagrees with the swept-up mass; the shell mass enters F_grav (SPEC-031), tau_IR (SPEC-027), and the EOM inertia, so the error propagates into all three.",
    "repro": "Add a test asserting |sum(4 pi r^2 rho dr) - M_sh|/M_sh < 1e-6 on the stored profile at several snapshots.",
    "severity": "S2",
    "confidence": "high"
  },
  {
    "id": "S8-C-10",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": "85",
    "class": "regime",
    "claim": "When the photon budget is exhausted inside the shell (ionisation-bounded), f_esc must be EXACTLY zero and a neutral outer layer must be integrated; when it is not exhausted (density-bounded), f_esc = Q(R_out)/Q_i > 0, there is no neutral layer, and T = T_ion throughout.",
    "evidence": "Photon conservation: photons that reach R_out escape by definition; photons that do not, do not. There is no third case. The two branches are what the boolean is_ionised argument selects.",
    "expected": "Two mutually exclusive, exhaustive branches; f_esc from the ODE state, never floored, clipped, or fitted.",
    "failure_scenario": "A floor/clip on f_esc (e.g. max(f_esc, 1e-6) or min(f_esc, 1-eps)) turns a physical zero into a small nonzero escape fraction, silently breaking the SPEC-028 budget closure and misreporting the headline escape-fraction result.",
    "repro": "Find a snapshot with a thick neutral layer and assert f_esc == 0.0 exactly; find a thin/dispersed snapshot and assert f_esc == Q_out/Q_i.",
    "severity": "S2",
    "confidence": "high"
  },
  {
    "id": "S8-C-11",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "37",
    "class": "units",
    "claim": "sigma_d (cm^2 per hydrogen NUCLEUS) must multiply the number density n_H, never the mass density rho; kappa_IR (cm^2 per GRAM) must multiply the mass column Sigma, never a number column.",
    "evidence": "sigma_d/(mu_H m_H) = 1.5e-21/(1.4*1.6735e-24) = 640 cm^2 g^-1, so the two constants differ by mu_H*m_H = 2.34e-24. SPEC-092 item 7 flags this as one of the top unit traps.",
    "expected": "tau_UV = sigma_d * N_H (dimensionless); tau_IR = kappa_IR * M_sh/(4 pi R2^2). Consistency anchor: kappa_UV/kappa_IR ~ 160, so a shell is UV-thick 160x earlier than IR-thick.",
    "failure_scenario": "Multiplying sigma_d by rho gives tau smaller by 2.34e-24: f_abs ~ 0, f_esc ~ 1, no radiation force. All values remain finite and superficially plausible, so this fails SILENTLY rather than loudly. (The reverse swap is a 4e23 error and would fail loudly.)",
    "repro": "Anchor test: n_H = 1e3 cm^-3 over 1 pc must give tau_UV = 1.5e-21*1e3*3.0857e18 = 4.63. Sigma = 0.25 g cm^-2 must give tau_IR = 1.0 for kappa_IR = 4.",
    "severity": "S1",
    "confidence": "high"
  },
  {
    "id": "S8-C-12",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "37",
    "class": "units",
    "claim": "Every product mixing a cgs micro-physics constant (alpha_B, sigma_d, sigma_HI, k_B, c, m_H) with an AU dynamical quantity (r in pc, Q in Myr^-1, L in Msun pc^2 Myr^-3, P in Msun pc^-1 Myr^-2) must be converted into one system before the RHS is formed.",
    "evidence": "This module is the single largest cgs/AU boundary in the code. Conversions [computed]: alpha_B = 2.59e-13 cm^3 s^-1 = 2.78e-55 pc^3 Myr^-1; sigma_d = 1.5e-21 cm^2 = 1.58e-58 pc^2; 1 cm^-3 = 2.938e55 pc^-3; 1 pc = 3.0857e18 cm.",
    "expected": "A single declared unit system inside get_shellODE, with all constants pre-converted at the boundary.",
    "failure_scenario": "Omitting the pc->cm factor in tau = sigma_d*n*dr makes tau smaller by 3.09e18, so f_abs ~ 0 and f_esc ~ 1 -- again finite and plausible. Omitting it in the recombination integral makes the photon budget close on the wrong length scale, moving R_IF by orders of magnitude.",
    "repro": "Dimensional unit test on get_shellODE: feed a hand-computed state with known cgs answer (Delta_R_ion = Q/(4 pi R2^2 alpha_B chi_e n^2) = 1.17e14 cm for Q=1e49, R2=5 pc, n=1e4) and compare.",
    "severity": "S1",
    "confidence": "high"
  },
  {
    "id": "S8-C-13",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": "85",
    "class": "coefficient",
    "claim": "The ionisation-front condition must be the shell-annulus balance Q_i = integral(4 pi r^2 [alpha_B chi_e n^2 + sigma_d n Phi] dr, R2..R_IF); the uniform-density filled-sphere Stroemgren radius R_St = (3 Q/(4 pi alpha_B chi_e n^2))^(1/3) must never be used as the layer thickness.",
    "evidence": "Delta_R_ion = R_St^3/(3 R2^2), so Delta_R_ion/R_St = (R_St/R2)^2/3. For R_St = 0.14 pc (n = 1e4, Q = 1e49) and R2 = 5 pc this is 2.7e-4, i.e. R_St over-states the ionised path length by ~3700x. SPEC-029's (R_i^3 - R_in^3) annulus form is the correct one.",
    "expected": "The annulus/integral form, reducing exactly to SPEC-029 in the uniform-n dust-free limit.",
    "failure_scenario": "Using bare R_St as a thickness over-predicts the ionised mass and the LyC dust path length (f_dust grossly over-predicted), can push Delta_R above R2 (violating the thin-shell assumption of SPEC-020), and mis-places the neutral layer.",
    "repro": "Free unit test: dust off, radiation off, uniform n; assert the solved R_IF satisfies Q_i = (4 pi/3) alpha_B * 1.1 * n^2 * (R_IF^3 - R2^3) to machine precision.",
    "severity": "S1",
    "confidence": "high"
  },
  {
    "id": "S8-C-14",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "37",
    "class": "regime",
    "claim": "A boolean is_ionised (sharp front, x = 1 inside / 0 outside) is valid only while ell_IF/Delta_R_ion = 4 pi R2^2 alpha_B chi_e n_H/(sigma_HI Q_i) << 1; the code must flag or handle the regime where it is not.",
    "evidence": "[computed] with sigma_HI = 6.3e-18 cm^2, Q_i = 1e49 s^-1, R2 = 5 pc: the ratio is 0.14 at n_H = 1e4 and 1.4 at n_H = 1e5. The TRINITY default nCore = 1e5 cm^-3 (SPEC-003) means compressed shells reach and exceed n_H ~ 1e5.",
    "expected": "Either a continuous x(r) in the stiff regime, or an explicit validity check with a logged warning/flag.",
    "failure_scenario": "With x < 1 the true recombination rate is alpha_B chi_e x^2 n^2, i.e. SLOWER, so the true ionised layer is thicker by ~1/x^2. Assuming x = 1 over-consumes the photon budget, under-predicts Delta_R_ion and f_esc. The bias has a fixed sign and grows with density, so it is worst exactly in the dense-cloud runs the paper features.",
    "repro": "Evaluate 4 pi R2^2 alpha_B chi_e n_H/(sigma_HI Q_i) from stored snapshots of a high-nCore run and report the fraction of snapshots where it exceeds 0.3.",
    "severity": "S2",
    "confidence": "medium"
  },
  {
    "id": "S8-C-15",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": "85",
    "class": "coefficient",
    "claim": "The reprocessed-IR force must be applied to the ABSORBED luminosity: F_rad = (L_bol/c)(1 - exp(-tau_UV))(1 + tau_IR), not (L_bol/c)(1 - exp(-tau_UV) + tau_IR).",
    "evidence": "Only luminosity actually absorbed by dust can be re-emitted in the IR and do work again. The additive form leaves a residual tau_IR*L/c as tau_UV -> 0, i.e. reprocessing of light that was never absorbed.",
    "expected": "The IR term vanishes as tau_UV -> 0 and reduces to tau_IR*L/c when tau_UV >> 1.",
    "failure_scenario": "With kappa_UV/kappa_IR ~ 160 the two forms agree to <1% in the UV-thick regime, so the error only bites in thin/dissolving shells -- which is exactly the regime that sets the dissolution stopping criterion (SPEC-102) and the escape fraction.",
    "repro": "Evaluate both forms on the stored (tau_UV, tau_IR) history; report the maximum fractional difference and when it occurs relative to the dissolution trigger.",
    "severity": "S3",
    "confidence": "medium"
  },
  {
    "id": "S8-C-16",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": "85",
    "class": "coefficient",
    "claim": "Ionising-photon momentum must be counted once: either via <h nu_i> Q_i/c for the ionising band plus L_n/c for the non-ionising band, or via a single L_bol/c with one tau -- never both.",
    "evidence": "L_bol = L_i + L_n by construction (SPEC-074). Adding an explicit LyC momentum term on top of an L_bol-based direct-radiation force double-counts the ionising band.",
    "expected": "A single, exhaustive band decomposition; the total absorbed momentum must not exceed L_bol/c.",
    "failure_scenario": "f_i is typically 0.1-0.3 of L_bol for a young cluster, so double-counting inflates F_rad by 10-30% -- enough to shift the transition time and the dispersal-vs-recollapse outcome, while the force-fraction plot (which normalises to F_tot) would hide it.",
    "repro": "Assert F_rad_total <= L_bol/c * (1 + tau_IR) on every snapshot; check the band bookkeeping at the call site.",
    "severity": "S2",
    "confidence": "medium"
  },
  {
    "id": "S8-C-17",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "37",
    "class": "coefficient",
    "claim": "f_cover must enter exactly once. Physically consistent convention: the photon FLUX per unit covered area is unchanged (Phi(R2) = Q_i/(4 pi R2^2)) while the COLUMN is enhanced (Sigma_patch = M_sh/(4 pi R2^2 f_cover)); escape is then f_esc_total = (1 - f_cover) + f_cover*f_esc_patch.",
    "evidence": "Isotropic emission means holes do not change the surface brightness seen by the covered patch, only the fraction of photons that hit it. The swept mass, however, must fit into a smaller solid angle.",
    "expected": "f_cover appears in the mass/column accumulation and in the escape bookkeeping, not in the photon initial condition as well.",
    "failure_scenario": "Scaling Phi(R2) by f_cover AND enhancing the column by 1/f_cover applies the correction twice, giving an f_cover^2 dependence in tau and hence in f_abs and f_esc. With coverFraction = 1.0 as the default this is invisible in fiducial runs and only appears in the sweeps that set C_f < 1.",
    "repro": "Run the same config at f_cover = 1.0 and f_cover = 0.5 and check that tau_patch scales as 1/f_cover (not 1/f_cover^2) and that f_esc_total >= 1 - f_cover.",
    "severity": "S2",
    "confidence": "medium"
  },
  {
    "id": "S8-C-18",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": "85",
    "class": "state",
    "claim": "P_HII must not be computed from a density that was itself back-solved from P_drive, or max(P_b, P_HII) (SPEC-022) degenerates to a no-op.",
    "evidence": "Pressure continuity at the contact discontinuity gives n_H(R2) = P_drive/(psi_ion k_B T_ion). Feeding that same n_H(R2) into P_HII = psi_ion n_H(R2) k_B T_ion returns P_drive identically. The non-circular physical content of P_HII is that a fully-ionised (density-bounded) shell cannot be compressed below the pressure of its own 1e4 K gas -- i.e. P_HII is a FLOOR set by the Stroemgren balance over the available shell mass, not a value read back off the inner face.",
    "expected": "P_HII derived from an independently pinned geometry (Q_i, M_sh, R2), and demonstrably able to exceed P_b.",
    "failure_scenario": "If P_HII == P_drive identically, TRINITY's headline novelty over WARPFIELD (SPEC-022, audit priority #1 in SPEC index) is inert, and the P_HII branch of the max() never activates -- yet the code would still report a P_HII column.",
    "repro": "Scan dictionary.jsonl for any snapshot with P_HII > Pb; if none exists across the energy phase of several configs, the circularity is real.",
    "severity": "S1",
    "confidence": "medium"
  },
  {
    "id": "S8-C-19",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "32",
    "class": "numerical",
    "claim": "A uniform radial grid of _NSHELL_MAX points across the whole shell cannot resolve the ionised layer; the front location must be root-found, not grid-resolved.",
    "evidence": "[computed] The density jumps by ~200 at the front (S8-C-06), so the ionised and neutral sublayers differ in thickness by 1-2 orders of magnitude. Example: Q = 1e49, R2 = 5 pc, n = 1e4 gives Delta_R_ion = 1.17e14 cm = 3.8e-5 pc, while the shell as a whole is ~1e-3 pc thick. A uniform grid would place O(1) points inside the ionised layer.",
    "expected": "Geometric/adaptive spacing, or two separately integrated segments joined at a root-found R_IF (which is what an is_ionised ARGUMENT, rather than a state component, enables).",
    "failure_scenario": "R_IF, and hence f_esc and f_dust, become functions of _NSHELL_MAX rather than of physics. A grid-convergence test would show the headline escape fraction drifting with a numerical constant.",
    "repro": "Grid-convergence test: rerun a fixed config with _NSHELL_MAX doubled and quadrupled; f_esc, f_dust and Delta_R must be stable to <<1%.",
    "severity": "S2",
    "confidence": "medium"
  },
  {
    "id": "S8-C-20",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "37",
    "class": "units",
    "claim": "T and the composition factor psi = n_tot/n_H must be switched TOGETHER at the ionisation front: psi_ion = 2.2, psi_atom = 1.1, psi_mol = 0.6.",
    "evidence": "P = psi n_H k_B T, so the jump ratio (S8.8) depends on the product psi*T. SPEC-092 item 2 lists all four psi values and warns that picking the wrong one for a region is the failure mode.",
    "expected": "Both T and psi (equivalently mu) change at the front, consistently with mu_ion_shell / mu_atom / mu_mol in the schema.",
    "failure_scenario": "Switching T from 1e4 to 1e2 but leaving psi = 2.2 gives a density jump of 100 instead of 200 -- a factor-2 error in the neutral density, hence in shell_nMax, tau_UV and tau_IR. Choosing atomic where the gas is molecular is a further factor ~1.8.",
    "repro": "Composition test T11: assert n_tot/n_H equals 2.2 in the ionised segment and 1.1 (or 0.6) in the neutral segment wherever P = n k T is formed.",
    "severity": "S2",
    "confidence": "high"
  },
  {
    "id": "S8-C-21",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": "39",
    "class": "state",
    "claim": "ShellProperties must expose an explicit success/validity flag alongside the physics, and every downstream consumer (F_rad, P_drive, f_esc reporting, the dissolution check) must gate on it.",
    "evidence": "Sections 6: on integration failure the derived quantities are not merely inaccurate but biased with a fixed sign. A result object that carries only physics values cannot distinguish 'converged, f_esc = 0.9' from 'truncated, f_esc looks like 0.9'.",
    "expected": "A boolean (or status enum) on the dataclass, written into the snapshot, and counted in metadata.json's termination_debug (SPEC-105).",
    "failure_scenario": "Without the flag, a sweep silently mixes converged and truncated shells; the truncated ones systematically over-report f_esc and under-report F_rad, biasing the published grid in one direction.",
    "repro": "Inspect the ShellProperties fields and grep the consumers for a convergence gate; check whether any snapshot records a shell-integration status.",
    "severity": "S2",
    "confidence": "medium"
  },
  {
    "id": "S8-C-22",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": "85",
    "class": "regime",
    "claim": "The solved shell thickness must satisfy Delta_R/R2 << 1; when it does not, the thin-shell EOM (SPEC-020), the 4 pi R2^2 area factors, and the plane-parallel P_HII treatment are all invalid and the run must flag or terminate.",
    "evidence": "[computed] In the normal regime Delta_R/R2 ~ 1e-5 to 1e-3, so the approximation is excellent. But as the shell decompresses (P_drive falling, dissolution approaching) Delta_R grows without bound while R2 is fixed, and there is no internal mechanism stopping Delta_R from exceeding R2.",
    "expected": "An explicit thin-shell validity check (e.g. Delta_R/R2 < 0.1) that flags rather than silently continues.",
    "failure_scenario": "A shell with Delta_R > R2 has negative inner volume in any expression using (R_out^3 - R2^3) approximations, and its force budget (which assumes all forces act at R2) is wrong by O(1). This is exactly the late-time regime that decides dispersal vs re-collapse.",
    "repro": "Plot Delta_R/R2 over a full run of param/simple_cluster.param and of the hidens edge config; report the maximum and whether anything in the code reacts to it.",
    "severity": "S2",
    "confidence": "medium"
  },
  {
    "id": "S8-C-23",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "37",
    "class": "state",
    "claim": "Positivity and monotonicity guards must be assertions/failures, not clamps: n_H > 0, Q >= 0 and non-increasing, tau >= 0 and non-decreasing, x in [0,1].",
    "evidence": "Each of these is exact physics, not a numerical convenience. Q < 0 means photons were created; n <= 0 means negative mass; tau decreasing means negative opacity.",
    "expected": "Violations abort the integration and are reported; they are never repaired by max(x, 0), abs(), or np.nan_to_num.",
    "failure_scenario": "Clamping Q at zero converts a stiff-integration overshoot into a spurious ionisation front, producing a neutral layer where none exists and reporting f_esc = 0 for a shell that is actually density-bounded -- the exact opposite of the physical answer.",
    "repro": "Search the module for clamping idioms (max(, np.clip, np.abs, nan_to_num) applied to the state vector; check the stored profiles for exactly-zero or exactly-clipped values.",
    "severity": "S2",
    "confidence": "medium"
  },
  {
    "id": "S8-C-24",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": "85",
    "class": "citation",
    "claim": "Neglect of trapped Lyman-alpha radiation pressure is a known, defensible simplification and should be recorded as such, not silently absent.",
    "evidence": "Case B implies ~0.68 Lyman-alpha photons per recombination [recalled, medium]; resonant scattering can multiply their momentum deposition by a large factor at high column, potentially rivalling the direct force. Dust absorption of Lyman-alpha is what normally prevents runaway, so the omission is usually defensible -- but it is an omission with a physical magnitude, not a null term.",
    "expected": "A documented statement of the neglected term and its regime of validity (high dust-to-gas, moderate column).",
    "failure_scenario": "In dust-poor or very high-column regimes the omitted Lyman-alpha force is a real under-estimate of F_rad; without documentation a future user cannot tell whether the model is applicable.",
    "repro": "",
    "severity": "S4",
    "confidence": "low"
  },
  {
    "id": "S8-C-25",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": "37",
    "class": "units",
    "claim": "The dust cross-section sigma_d and the IR opacity kappa_IR must carry the SAME metallicity scaling, since they describe the same grain population.",
    "evidence": "SPEC-028 declares sigma_d = 1.5e-21 cm^2 (Z/Zsun); SPEC-027 flags that dust_noZ = 0.05 Zsun implies a Z scaling exists. Dust-to-gas ratio scales ~linearly with Z, so both opacities scale with Z. The consistency anchor kappa_UV/kappa_IR = sigma_d/(mu_H m_H kappa_IR) ~ 160 must be Z-independent.",
    "expected": "Both constants scaled by (Z/Zsun) (with the dust_noZ floor applied identically to both), so their ratio is invariant.",
    "failure_scenario": "Scaling only sigma_d leaves the UV and IR channels disagreeing about how much dust exists; at low Z the shell becomes UV-transparent while remaining IR-opaque, which is physically impossible and inverts the sign of the net radiation force's Z dependence.",
    "repro": "Compare sigma_d and kappa_IR at Z = 1 and (if the schema allowed it) Z = 0.1; assert the ratio sigma_d/(mu_H m_H kappa_IR) is unchanged.",
    "severity": "S3",
    "confidence": "medium"
  }
]
```
