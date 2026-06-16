# Pressure terms & the meaning of `n`: the `2nkT` / `mu_p`,`mu_n` audit

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
> harness/figure under `docs/dev/scratch/` as the hybr work did) — never left in `/tmp` or
> an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.
>
> **Audit status (2026-06-08):** this file is **already self-declared SUPERSEDED**
> (banner below) and remains accurate **as a historical record** — its pointer to
> `n-consistency-audit.md` / `n-consistency-implementation-plan.md` is valid and
> its μ constants are correct. No content fix needed.

> **⚠️ SUPERSEDED (historical).** This first pass — written *before* the model
> paper fixed the convention — reached the *opposite* conclusion on the bubble
> (it treated `mu_ion` as the mass-conversion weight). The paper pins
> `n ≡ n_H`, so mass uses `mu_convert` (=μ_H) and the bubble factor is `μ_H/μ_p`,
> not 1. The authoritative record is **`n-consistency-audit.md`** (physics) and
> **`n-consistency-implementation-plan.md`** (what shipped). Kept only to show
> the reasoning that led there.

Single source of truth for the hand-wavy pressure conversions in TRINITY —
the `P = 2nkT` factors and the `mu_p/mu_n` ratios — and whether they are
internally consistent with the **doubly-ionised helium** composition the
mean-molecular-weight constants already encode. Same shape as the sibling
audits: **Part I** is the audit (*what is*); **Part II** is the fix posture
and the exact, turnkey patch.

> **Status (2026-06): audit only. No runtime numbers moved.**
> The headline item (Finding A, the bubble `P/(2k_BT)`) is **behavior-affecting**:
> it changes bubble mass, self-gravity, and cooling luminosity. It is therefore
> **held for model-author sign-off** (same posture as the grid fix in
> `bubble-integrator-robustness.md`). Part II gives the precise patch so the
> decision is one step.

## TL;DR

- **The μ constants are fine.** `mu_atom=14/11`, `mu_ion=14/23`,
  `mu_mol=14/6`, `mu_convert=1.4` (`_input/registry.py:313-316`) correctly
  encode He:H = 1:10 by number with **doubly-ionised** He (H⁺ + He⁺⁺ ⇒
  10 p + 10 e + 1 α + 2 e = **23** particles). The bug is *not* the weights.
- **The bug is that "`n`" means three different things** in three modules,
  and the bubble's `P = 2nkT` (a *pure-hydrogen* relation) collides with the
  He-aware `mu_ion`.
- **Finding A (real, factor-of-2):** the bubble computes
  `n = Pb/(2·k_B·T)` then `rho = n·mu_ion`. Those two are mutually exclusive.
  Result: bubble **mass and self-gravity are ≈2× too low**, and the `n` fed
  to cooling is 2× off from `n_tot`.
- **Finding B:** the non-CIE cooling table's density axis is **`n_H`**
  (documented in `cooling-refactor-audit.md:521`; the opiate tables even
  carry a separate `nedens` column). The bubble feeds it `≈1.15·n_H`.
- **Finding C (dead code):** `get_shellParams.py:30` has the μ-ratio
  **inverted** vs. the live formula in `shell_structure.py:115`. The function
  is never imported — latent landmine only.
- **Finding D (OK):** the shell `mu_p/mu_n` ODE *is* self-consistent under
  `nShell ≡ ρ/(mu_atom·m_H)`, and respects doubly-ionised He.
- **Finding E:** no single definition of `n` survives the module handoffs
  (cloud `n_H` → shell `ρ/(mu_atom m_H)` → bubble `P/(2k_BT)`), and the
  boundaries don't convert.

---

# Part I — Audit (what is)

## I.0 Composition baseline (the part that is correct)

He:H = 1:10 by number. Per 10 H + 1 He, total mass = 10·1 + 1·4 = **14 m_H**.

| constant | value | counting | particles per (10H+1He) |
|---|---|---|---|
| `mu_atom` | 14/11 ≈ 1.273 | neutral atomic | 10 H + 1 He = **11** |
| `mu_ion`  | 14/23 ≈ 0.609 | **doubly-ionised** H⁺+He⁺⁺ | 10 p + 10 e + 1 α + 2 e = **23** |
| `mu_mol`  | 14/6  ≈ 2.333 | molecular | 5 H₂ + 1 He = **6** |
| `mu_convert` | 1.4 | mass per H nucleus | = 14/10 |

`mu_ion` is the **mean mass per total particle**, and it *does* count He's two
electrons. The ratios that matter downstream:

```
n_tot / n_H = 23/10 = 2.3          (= mu_convert / mu_ion = 1.4 / (14/23))
n_e   / n_H = 12/10 = 1.2
n_tot = ρ/(mu_ion·m_H) = P/(k_B T)   ← NO factor of 2 for this gas
```

The pure-hydrogen identity `P = 2 n_H k_B T` (because then `n_e=n_p=n_H`,
`n_tot=2 n_H`) is exactly the "`2`" that appears in the bubble. For *this*
composition the He-correct prefactor is **2.3**, not 2 — or, equivalently,
just use `n_tot = P/(k_B T)` with no prefactor.

## I.1 Finding A — bubble `n = P/(2k_BT)` vs `ρ = n·mu_ion` (factor of 2) — **REAL**

`bubble_structure/bubble_luminosity.py` computes the density five times as

```python
n = Pb / (2 * params['k_B'].value * T)     # lines 338, 390, 465, 498, 948
```

and converts to mass density once, in `_get_mass_and_grav`:

```python
rho_new = n[::-1] * params['mu_ion'].value  # line 970
```

`P = 2nkT` assumes `n_e = n_ion = n` (pure H). `mu_ion = 14/23` is mass
*per total particle*. For `rho = n·mu_ion·m_H` to be correct, `n` **must** be
`n_tot = P/(k_B T)` — no factor of 2. The two assumptions cannot both hold.

The rest of the *same file* silently uses the no-factor-of-2 convention
`P = ρ k_B T/(mu_ion m_H)`:

- `_get_init_dMdt` (Weaver+77 Eq. 33): `… * mu_ion / k_B …` (line 595)
- `_get_bubble_ODE_initial_conditions`: `constant = 25/4 * k_B / mu_ion …`
  (line 927) and `… * k_B * T / mu_ion / Pb` (line 933)

so `mu_ion` there is the mean-mass-per-particle that pairs with
`n_tot = P/(k_B T)`. The `/2` in `n_array` contradicts the conduction physics
sitting right next to it.

**Consequences**
- `rho_new`, the cumulative `mBubble` (`bubble_luminosity.py:562-563`,
  fed back as `bubble_mass`), and the self-gravity from `_get_mass_and_grav`
  are **a factor of 2 too low**.
- The `n` handed to `net_coolingcurve.get_dudt` and to `n_bubble**2 * Λ_CIE`
  is `n_tot/2`, i.e. 2× off from the total density and ~15% off from `n_H`
  (see Finding B).

## I.2 Finding B — the cooling table wants `n_H`

The non-CIE (CLOUDY/opiate) table's density axis is **hydrogen density `n_H`**:

- `docs/dev/cooling-refactor-audit.md:521` —
  `cool_col_nonCIE_ndens → n_H [cm^-3]`.
- the opiate tables (`lib/default/opiate/opiate_cooling_*.dat`) carry both an
  `ndens` *input* column and a separate `nedens` (electron density) column —
  i.e. `ndens` is the CLOUDY `hden = n_H`, not a total or electron density.

CIE cooling uses `dudt = ndens**2 * Λ_CIE` (`net_coolingcurve.py:126`,
`bubble_luminosity.py:411`). The bubble feeds `Pb/(2k_BT) ≈ 1.15·n_H` into an
axis that expects `n_H`. (The CIE `Λ` normalisation of the bundled curves —
`n_H²` vs `n_e n_H` vs `n_t²` — is a *separate*, pre-existing question and is
out of scope here.)

## I.3 Finding C — `get_shellParams.py:30` inverted (dead code)

```python
# get_shellParams.py:30  — WRONG (reciprocal): mu_atom/mu_ion
nShell0 = params['mu_atom'].value/params['mu_ion'].value /(k_B*TShell_ion) * Pb
# shell_structure.py:115 — CORRECT and actually used: mu_ion/mu_atom
nShell0 = params['mu_ion'].value /params['mu_atom'].value /(k_B*TShell_ion) * Pb
```

`get_nShell0()` is **never imported or called** (grep: only its own
definition). It has zero runtime effect, but if resurrected it is the
reciprocal of the correct value (≈ (23/11)² ≈ 4.4× off in `nShell0`). Flagged,
not changed (pre-existing dead code).

## I.4 Finding D — shell `mu_p/mu_n` ODE is self-consistent — **OK**

`shell_structure/get_shellODE.py:93` (`dndr = mu_p/mu_n/(k_B·t_ion)·…`) and
`:118` (factor 1 in the neutral region) are consistent **iff**
`nShell ≡ ρ/(mu_atom·m_H)`. That single convention holds throughout
`shell_structure.py`:

- inner-edge BC `nShell0 = (mu_ion/mu_atom)/(k_B T_ion)·Pb` (line 115) —
  this is `ρ_edge/(mu_atom m_H)` with `ρ_edge = Pb·mu_ion/(k_B T_ion)`. ✓
- mass `ρ = nShell·mu_atom` (lines 167, 253). ✓
- I-front jump `nShell0 ← nShell0·(mu_atom/mu_ion)·(T_ion/T_neu)` (line 298),
  which is exactly pressure continuity `n_tot,ion k T_ion = n_tot,neu k T_neu`. ✓

So the shell respects doubly-ionised He. **Caveat (minor, ~10%, not 2×):** the
recombination term `nShell**2·α_B` (`get_shellODE.py:95`) and the Strömgren
density (`shell_structure.py:237`) use this `nShell` as if it were the electron
density `n_e`, but `nShell = ρ/(mu_atom m_H) ≈ (11/12) n_e`. Noted for
completeness; separable from the pressure-term fix.

## I.5 Finding E — three incompatible definitions of `n`

| module | `n` means | `ρ` from `n` | status |
|---|---|---|---|
| cloud / ISM (`mass_profile.py`, `get_InitCloudProp.py`, BE/PL spheres, `get_InitPhaseParam.py:146`) | `n_H` | `n_H·mu_convert` (1.4) | clean, He-correct, documented; the *de-facto reference* |
| shell (`shell_structure.py`, `get_shellODE.py`) | `ρ/(mu_atom·m_H)` (≈1.1·n_H) | `n·mu_atom` | self-consistent (Finding D) |
| bubble (`bubble_luminosity.py`) | `P/(2k_BT)` (≈1.15·n_H) | `n·mu_ion` | inconsistent (Finding A) |

`registry.py:296-297` is explicit that `nCore`/`nISM` are `n_H`. The handoffs
(`Pb` → shell `nShell0`; ambient `n_H` → bubble) cross conventions without
converting.

---

# Part II — Fix posture (what to do)

**Behavior-affecting. Held for model-author sign-off.** The recommendation is
to standardise the bubble on **`n_H`** — the convention the cloud module and
the cooling table already use — rather than on `n_tot`. Rationale: dropping the
`/2` alone (→ `n_tot`) fixes the *mass* but makes the *cooling* worse (it would
feed `n_tot = 2.3·n_H` into an `n_H` axis, vs. `1.15·n_H` today). The `n_H`
convention fixes **both** mass and cooling and unifies the whole code on one
meaning of `n`.

## II.1 The patch (bubble = `n_H`)

In `bubble_luminosity.py`, replace the five density definitions

```python
n = Pb / (2 * params['k_B'].value * T)
```

with the He-correct hydrogen density (prefactor `n_tot/n_H = mu_convert/mu_ion = 2.3`):

```python
# n_H from total pressure: n_tot = Pb/(k_B T); n_H = n_tot · mu_ion/mu_convert
n = Pb * params['mu_ion'].value / (params['mu_convert'].value * params['k_B'].value * T)
```

and in `_get_mass_and_grav` (line 970) change the mass conversion to the
state-independent mass-per-H weight, matching the cloud module:

```python
rho_new = n[::-1] * params['mu_convert'].value   # was: * mu_ion
```

These two changes together give `ρ = Pb·mu_ion/(k_B T)·m_H = n_tot·mu_ion·m_H`
(the correct total mass density) **and** hand the cooling table a true `n_H`.

*Minimal alternative (not recommended):* delete only the `2` (→ `n_tot`,
`ρ = n·mu_ion` unchanged). Fixes mass, leaves the cooling axis 2.3× off.

## II.2 Expected numeric shifts (so the diff is explainable)

- bubble `n` for cooling: `×0.87` (`1.15·n_H → 1.0·n_H`) ⇒ CIE `n²Λ` ≈ `×0.76`.
- bubble `ρ`, `mBubble`, self-gravity: `×2.0` (was a factor-of-2 low).
- everything keyed off `mu_ion`-as-mean-mass-per-particle (Weaver `dMdt`,
  ODE init) is **unchanged**.

## II.3 Verification plan

1. Unit test: assert `ρ = n·mu_convert` and `n = Pb·mu_ion/(mu_convert·k_B·T)`
   reproduce `ρ = Pb·mu_ion/(k_B·T)·m_H` to round-off (one closed-form check).
2. Golden-output diff on a quickstart param: confirm the shifts in II.2 and
   that nothing *else* moves (mass ×2, cooling ×0.76 — no third change).
3. Confirm `bubble_mass` feeding the shell BC stays physical.

## II.4 Out of scope (separate tickets)

- Finding C: fix or delete the dead `get_shellParams.get_nShell0`.
- Finding D caveat: `nShell` vs `n_e` (~10%) in recombination/Strömgren.
- CIE `Λ` normalisation (`n_H²` vs `n_e n_H`) of the bundled curves.
