# κ_mix implementation + units spec — step 1 of the gated Rung-B wiring (DESIGN, no production code)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time design/plan, not a maintained spec; the code moves faster than
> these notes (paths, line numbers, "what shipped" status drift). **Any agent or
> person reading this: treat it as unverified. Re-check each claim, snippet, and
> line reference against current source before relying on it.**
>
> 🔄 **Living plan — recheck and refine on every visit.** Re-verify the claims and
> line references above against current source; update drift; rethink the strategy
> itself (a cleaner injection, a tighter gate) and note what changed and why (date
> it). Leave it better than you found it. **Keep all banner paragraphs at the top.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The offline go/no-go that
> justifies this wiring is the committed `data/make_kmix_prototype.py` +
> `data/kmix_prototype.{csv,png}` (reproducible without sims). Any equivalence
> artifact added here must be committed with its exact command.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** Siblings:
> `INDEX.md`, `PLAN.md`, `KMIX_PROTOTYPE.md` (the offline GO), `KMIX_DIFFUSIVITY.md`
> (where λδv comes from), `RUNGB_SCOPING.md` §8 (the gated-production end state),
> `F_KAPPA_FUNCTIONAL_FORM.md` §13 (why κ_mix, not a scalar). Reconcile any
> number/claim that disagrees; never update one in isolation.

---

> **⏸ SHELVED (2026-06-30): the direct κ_mix-into-the-Weaver-ODE injection this spec designs is superseded by
> the El-Badry θ_target path (`ELBADRY_REFERENCE.md`).** The full-paper read showed El-Badry's θ ≡ TRINITY's
> trigger θ and is available as a 3D-calibrated closed form (Eq 37/38), so we can impose it via the existing
> gated `cooling_boost_mode='theta_target'` without porting κ_mix into the structure solver (which saturated and
> went numerically unstable). This spec is retained for the **dimensionless-multiplier units strategy** (§2,
> still correct and reusable) and as the record of *why the direct port was abandoned*. Two corrections from the
> full read: **Parker conductivity is negligible** (κ_P ≪ κ_S) and **saturation barely affects θ** (it mostly
> changes Mhot ~15-20% and early-time numerics) — so neither is the missing piece I speculated last turn; the
> real issue was that the closed-form θ makes the whole port unnecessary.

## 0. What this is, and the guardrail

This is the **design + units spec** for wiring El-Badry's mixing conductivity κ_mix into TRINITY's
bubble-structure solver — written as a PLAN. **It changes no production code.** It is the bridge between the
offline prototype (`KMIX_PROTOTYPE.md`, GO) and the next executable step (the *self-consistent offline*
re-solve), and finally the gated production mode (`RUNGB_SCOPING.md` §8). Under the maintainer's standing
guardrail, nothing here reaches the production path until it is **tested on all 8 configs with units handled**
and proven **byte-identical with the mode off**.

## 1. Why κ_mix and not the scalar f_κ (one paragraph, carried from §13)

`cooling_boost_kappa` (f_κ) multiplies the Spitzer coefficient: `κ = f_κ·C_th·T^(5/2)`. Because Spitzer
`∝ T^(5/2)` **vanishes in the cool layer** (2–4×10⁴ K) where the mixing-layer cooling actually lives, no
*scalar* f_κ can put conductivity *there* without exploding it in the hot interior — and worse, f_κ *raises*
the evaporative mass flux (dMdt ∝ κ^(2/7)), the very El-Badry coupling a faithful term must *suppress*. The
faithful fix is El-Badry's **temperature-independent** mixing conductivity added as a floor:

```
κ_eff(T) = max( κ_mix , κ_Spitzer(T) ) ,   κ_Spitzer = C_th·T^(5/2) ,   κ_mix = (λδv)·n·k_B/(μ m_p)
```

κ_mix is flat in T, so it dominates exactly the cool layer and is negligible in the hot interior
(`T_cross ≈ ((λδv)·Pb/C_th)^(2/7) ≈ 2.4–5.0×10⁶ K`, far above the layer — `KMIX_PROTOTYPE.md` §2).

## 2. The units strategy — implement κ_mix as a DIMENSIONLESS multiplier on the existing Spitzer term

**This is the single most important design decision and it neutralizes the recurring units bug class.**

TRINITY's bubble ODE mixes unit systems on purpose: `C_thermal` is **cgs** (`erg s⁻¹ cm⁻¹ K⁻⁷ᐟ²`), `Pb` is
**AU** (`Msun Myr⁻² pc⁻¹`), `T` is K, and `r` is pc. The existing RHS term `Pb / (C_th·T^(5/2)) · (…)` is
internally consistent *as written* (the code is validated against `cooling_boost_kappa=1`). **If we add κ_mix
in raw units we re-enter exactly the AU/cgs trap.** So we do not add it raw. Instead:

> Form the **dimensionless ratio** `R ≡ κ_mix / κ_Spitzer`, computed **entirely in cgs**, and scale the
> existing Spitzer term by `max(1, R)`:
> ```
> κ_eff = κ_Spitzer · max(1, R) ,    R = (λδv)·Pb_cgs / (C_th·T^(7/2))         [dimensionless]
> ```
> Substituting `n = Pb/(k_B T)` (pressure equilibrium at the front) makes `k_B` and `μ m_p` cancel, leaving
> the clean ratio the prototype already validated (`KMIX_PROTOTYPE.md` §1, dimensional self-check printed).

Because `max(1, R)` is a pure number, every existing mixed-unit expression stays dimensionally exactly as it
is — we only multiply a denominator by a dimensionless factor. **No AU↔cgs conversion ever enters the solver
RHS.** The only conversion is the one *inside* R, isolated and unit-tested:

- `Pb_cgs = Pb_au / 1.5454414956718×10¹²` (`Pb_cgs2au` in `trinity/_functions/unit_conversions.py`).
- `λδv`: input in `pc·km/s` → `cm²/s` via `× 3.086×10²³` (one conversion, at read time).
- `C_th = 6×10⁻⁷` cgs (registry), `T` in K. R is dimensionless.

**Byte-identical-off falls out for free:** with the mode off (`λδv = 0`), `R = 0`, `max(1, 0) = 1`,
`κ_eff = κ_Spitzer` *exactly* — the multiplier is the literal float `1.0`, so the arithmetic is unchanged bit
for bit, not merely close.

## 3. The three conduction sites (verified `bubble_structure/bubble_luminosity.py`, 2026-06-30)

| site | line | role | needs κ_eff? |
|---|---|---|---|
| `_get_init_dMdt` | **291** | `C_th = cooling_boost_kappa·C_thermal`; dMdt **seed** ∝ `(t·C_th/R2²)^(2/7)·Pb^(5/7)` | **No** — seed only |
| `_get_bubble_ODE_initial_conditions` | **370** | boundary at `T_init = _T_INIT_BOUNDARY = 3×10⁴ K`; `constant = 25/4·k_B/μ_ion/C_th` | **Yes** |
| `_get_bubble_ODE` | **406** | RHS: `dTdrr = Pb/(C_th·T^(5/2))·(…) − 2.5·dTdr²/T − 2·dTdr/r` | **Yes** |

Key finding that drives the scope: **`_T_INIT_BOUNDARY = 3×10⁴ K`** (`bubble_luminosity.py:52`). The
integration's outer boundary sits *inside* the cool, κ_mix-dominated layer (well below `T_cross ≈ 5×10⁶ K`),
**not** in the Spitzer regime — so the boundary condition (:370) must use κ_eff too, not just the RHS (:406).
The integration sweeps from that cool boundary inward to the hot interior, crossing `T_cross` mid-domain.

- **Site :291 — leave Spitzer.** This is only the **initial guess** for the dMdt root-find; the converged dMdt
  is fixed by the ODE+BCs (which *do* use κ_eff). A Spitzer seed lands in the basin and keeps the diff
  minimal. (If convergence ever degrades in the κ_mix regime, upgrade the seed — flag, don't pre-optimize.)
- **Site :370 — ⚠️ CORRECTED by the self-consistent test (`KMIX_SELFCONSISTENT.md` §2).** A naive
  `C_th → C_th·max(1, R(T_init, Pb))` here **DIVERGES**: the Spitzer boundary offset `dR2 ∝ C_th`, so the
  multiplier scales `dR2` by `R(T_init) ≫ 1`, pushing `r2_prime = R2−dR2` past `R1` (invalid domain).
  Patching both :370 and :406 failed at *every* λδv>0; **RHS-only (:406, IC kept Spitzer) is stable.** So the
  boundary must either **stay Spitzer** (the validated scoping choice) or get a κ_mix-specific layer
  re-derivation (the El-Badry layer ≠ the `dR2 ∝ 1/C` closure) — NOT a naive C-scaling.
- **Site :406 — general-κ RHS (the substantive change).** Two pieces:
  1. **Prefactor:** `Pb/(C_th·T^(5/2))` → `Pb/(C_th·T^(5/2)·max(1, R))` = `Pb/κ_eff`.
  2. **The κ′ term — ⚠️ CORRECTED (`KMIX_SELFCONSISTENT.md` §2b).** The `−2.5·dTdr²/T` term is the
     `(dκ/dT)/κ·(dTdr)²` contribution. **κ_mix is NOT flat in T:** `κ_mix = (λδv)·n·k_B ∝ n ∝ 1/T` at fixed
     Pb, so `(dκ_mix/dT)/κ_mix = −1/T`, *not* 0. The hard-max form below (used in the first harness) is
     therefore wrong in the κ_mix regime **and** numerically too stiff (it NaNs the early high-Pb epochs). Use
     the **smooth-max** instead — `κ_eff = κ_S·(1+R^s)^(1/s)` (s≈4) — which is C¹ and gives the correct kprime
     analytically:
     ```python
     # smooth-max (preferred): R = kappa_mix/kappa_Spitzer ∝ 1/T^3.5
     blend = (1.0 + R**s) ** (1.0 / s)            # kappa_eff = kappa_Spitzer * blend
     kprime_over_k = (2.5 - 3.5 * R**s / (1.0 + R**s)) / T   # -> 2.5/T (Spitzer), -> -1/T (kappa_mix)
     dTdrr = (Pb / (C_th * T**2.5 * blend) * (...) - kprime_over_k * dTdr**2 - 2 * dTdr / r)
     ```
     For bit-identical-off, branch λδv=0 → the verbatim production expression (the `*blend`=1.0 / kprime
     reorder otherwise breaks G1, `KMIX_SELFCONSISTENT.md` §1). The earlier hard-max+0 form is retained only as
     the first-pass harness; **do not ship it.**
  **Verify against Weaver+77 Eqs 42–43 during implementation** that *only* the `−2.5·dTdr²/T` term carries the
  κ-derivative — the other `2.5`/`5/2` factors in the bracket are the enthalpy `γ/(γ−1)` (physical, κ-independent)
  and must **not** be switched. This mapping is asserted here as the design; it is a **gate item**, not a
  proven fact, until the per-call equivalence test (4.1) confirms it.
- **Numerical note (crossover kink):** `max(κ_mix, κ_S)` is C⁰ but not C¹ at `T_cross`; `kprime_over_k`
  jumps. If the integrator chatters there, replace the hard max with a smooth blend
  `κ_eff = (κ_mix^s + κ_S^s)^(1/s)` (s≈4–8) and the matching analytic `(dκ_eff/dT)/κ_eff`. Start with the hard
  max (simplest, exact-off); switch only if a config shows step-size collapse near `T_cross`.

## 4. The gate parameter (mirror the existing `cooling_boost_mode` pattern)

Add two registry params (`trinity/_input/registry.py`, alongside `cooling_boost_*` at :349–352), both
`category='input_solver'`, both defaulting to the **off / byte-identical** value:

| param | default | meaning |
|---|---|---|
| `kappa_mix_mode` | `'none'` | `'none'` = κ_eff = κ_Spitzer, **byte-identical**. `'elbadry'` = `κ_eff = max(κ_mix, κ_Spitzer)`. |
| `kappa_mix_lambda_dv` | `0.0` | λδv in **pc·km/s** (input convention). The single magnitude knob. Default 0 ⟹ R=0 ⟹ off even if mode flipped. |

Double off-switch by design (mirrors `cooling_boost_mode='none'` + `cooling_boost_fmix=1.0`): the mode string
gates the code path; λδv=0 makes the path a no-op even when on. Both must be `run_const=True`,
`exclude_from_snapshot=True` (run constants, like the cooling-boost family). `info` strings must point here and
state "default = BYTE-IDENTICAL", matching the house style of `cooling_boost_kappa`'s registry note.

**⚠️ UPDATE (`KMIX_SELFCONSISTENT.md` §2): the "calibrate λδv to Lancaster" plan is RETIRED.** The
self-consistent solve shows θ **saturates by λδv ≈ 0.01** (κ_mix swamps Spitzer at tiny λδv), so λδv is **not a
continuous knob** — the κ_mix-saturated θ is a fixed per-config output, and it misses the Lancaster band for
mid/dense clouds. λδv as a gate param still works as an **on/off** (0 = byte-identical; any small >0 = κ_mix
floor on), but it cannot be *dialed* to a target. Whether to wire κ_mix at all now depends on the strategy
revision (`KMIX_SELFCONSISTENT.md` §3), so the registry params below are **on hold**, not yet added.

## 5. Equivalence gates (CLAUDE.md rule 5 — this is an iterative/solver path)

Per the planning ladder, a per-call check is **necessary but NOT sufficient** for a solver edit; clear it with
a full-run gate on the stiff edges in **separate processes** at **matched `t`**.

### 4.1 Per-call (cheap, necessary): mode-off is bit-identical, mode-on matches the offline ratio
- With `kappa_mix_mode='none'`: call `_get_bubble_ODE` / `_get_bubble_ODE_initial_conditions` on saved
  states; assert **bit-identical** outputs vs current code (the multiplier is literally `1.0`).
- With mode on: assert the injected `R` at a few (T, Pb) equals `data/make_kmix_prototype.py`'s ratio for the
  same inputs (units cross-check, ties the solver to the validated offline formula).

### 4.2 Full-run, all 8 configs, separate processes, matched `t` (the real gate)
The 8 configs (`INDEX.md` §3): `simple_cluster`, `midrange_pl0`, `be_sphere`, `pl2_steep`,
`large_diffuse_lowsfe`, `small_dense_highsfe`, `fail_repro` (heavy 5e9), `small_1e6` (control).
- **Mode OFF:** every one of the 8 must produce a **byte-identical `dictionary.jsonl`** vs `git show HEAD`
  (diff + content hash). This is the byte-identical-off proof, on the full set, not a spot check.
- **Mode ON (calibration run):** runs complete (no integrator failure at `T_cross`), θ moves toward the
  Lancaster band, and the stop fate stays sane. Record θ(config, λδv) as the calibration table.
- Stiff edges to stress first: `small_dense_highsfe` + `f1edge_{lowdens,hidens}` (feedback strength × density).

### 4.3 Self-consistent offline FIRST (before any production wiring)
The very next executable step is **not** the registry edit — it is a harness that **calls** the
`bubble_luminosity.py` solver functions with κ_mix injected (monkeypatched / a thin wrapper), re-solving the
structure off the production path, on all 8 configs, to confirm θ behaves and the integrator is stable across
`T_cross`. Only after that passes do the registry params + gated branch land (`RUNGB_SCOPING.md` §8). Order:
**offline self-consistent (4.3) → per-call (4.1) → gated full-run (4.2) → production default stays `none`.**

## 6. Apply order (when green-lit; production default stays `none`)
1. Build the self-consistent offline harness (4.3); pass on 8 configs; commit its CSV/figure under `data/`.
2. Add `kappa_mix_mode`/`kappa_mix_lambda_dv` to the registry (default off).
3. Implement the dimensionless-multiplier κ_eff at sites :370 and :406 (seed :291 unchanged); the
   `kprime_over_k` switch; the isolated `R` helper with its one λδv and Pb_cgs conversion + a units unit-test.
4. Gate 4.1 (bit-identical off) → 4.2 (8-config full-run, byte-identical off + calibration on) → full `pytest`
   + ruff F-rules.
5. Commit to `feature/PdV-trigger-term-pt2`; **production default `kappa_mix_mode='none'` — no behavior change
   ships.** Reconcile `INDEX.md` §3 track + `PLAN.md` ledger.

*Written 2026-06-30 on `feature/PdV-trigger-term-pt2`. No production code touched; design only.*
