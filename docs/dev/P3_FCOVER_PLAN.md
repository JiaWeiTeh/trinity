# P3 — Photon covering fraction (`f_cover`): detailed plan

> Companion to `LEAKING_LUMINOSITIES_SKELETON.md` (problem **P3** in the code review).
> **Status: PLAN ONLY — nothing implemented.** Branch when built: same `feature/add-Cf`
> lineage; core-radiation change → Joel signs off before merge.
> Line numbers drift — locate by file + function + quoted string.

## 1. Goal & scope boundary

Couple the shell's **in-run photon covering fraction** `f_cover` to the leak geometry
(`coverFraction`, or the breakout-triggered `Cf_eff(R2)` once that exists), and apply it
**consistently**. Today `f_cover` is hard-coded to `1` and applied asymmetrically.

**In scope (P3):** the *dynamical* effect of holes on the **dust / bolometric** radiation —
shell optical depth → absorbed fraction → `F_rad` (radiation-pressure force + IR trapping).

**Explicitly NOT in scope:**
- **Ionising LyC escape budget** (`fleak = 1−Cf` for `Q0`): that is the CLOUDY
  post-processing term — **Phase G**, not P3. (See §6 for the deliberate inconsistency this creates.)
- **Mass sink** (P1) and **breakout trigger** (P2/STEP 3): separate tasks; P3 only *consumes*
  the covering fraction they define.

## 2. Current state (verified against source)

| Item | Location | Fact |
|---|---|---|
| `f_cover` value | `shell_structure/shell_structure.py:105-106` | `# TODO: Add f_cover from fragmentation mechanics` / `f_cover = 1` (hard-coded). |
| Passed to ODE | `shell_structure.py:158` (ionised), `~:317` (neutral) | `args=(f_cover, is_ionised, params)` — both call sites. |
| Applied (ionised) | `get_shellODE/get_shellODE.py:100` | `dtaudr = nShell * sigma_dust * f_cover` ✅ |
| **Applied (neutral)** | `get_shellODE.py:122` | `dtaudr = nShell * sigma_dust` ❌ **no `f_cover`** — the inconsistency. |
| Not applied to LyC | `get_shellODE.py:98` | `dphidr = … − nShell*sigma_dust*phi` — `f_cover` absent (ionising photons ignore holes). |

**Downstream of `tau`:**
`tau_rEnd` → `f_absorbed_neu = 1 − exp(−tau_rEnd)` (`shell_structure.py:390`) →
`f_absorbed = (f_absorbed_ion·Li + f_absorbed_neu·Ln)/(Li+Ln)` (`:391`) →
`shell_fAbsorbedWeightedTotal` (`:446`) →
`F_rad = fAbsWeighted · Lbol/c · (1 + tauKappaRatio·κ_IR)` (`energy_phase_ODEs.py:134-136`).
`tau` also feeds `neg_exp_tau` inside `dndr` (`get_shellODE.py:94,119`), so it perturbs the
**density structure** (hence `shell_mass`, `tau_kappa_IR`, `n_IF`) as well — not just `F_rad`.

**Net today:** with `f_cover ≡ 1` the asymmetry is a `×1` no-op (dormant). It becomes a real
bug only once `f_cover < 1` is introduced without fixing the neutral branch.

## 3. The physics — and the modelling fork

A covering fraction means the shell has holes over a sky fraction `(1−Cf)`. Two ways to put
that into a 1-D shell:

- **Model A — scale `dτ/dr` by `f_cover` (the existing scaffold; the review's P3).**
  Effective extinction `∝ exp(−Cf·τ_full)` for *every* ray. Simple, matches current code,
  but it is a **uniform dilution**, not holes — and because it lives inside the ODE it also
  changes the integrated density structure (`dndr` via `neg_exp_tau`), so its effect is
  **pervasive and hard to isolate** (shell mass, `tau_kappa_IR`, `n_IF` all shift).

- **Model B — split covered/uncovered at the absorbed-fraction level (recommended).**
  Leave the structure ODE at `f_cover = 1` (full physics), then apply holes to the *outputs*:
  `(1−Cf)` of sightlines are holes that absorb nothing, `Cf` see the full column. Because the
  weighting is linear,
  ```
  f_absorbed_eff      = Cf · f_absorbed          # → shell_fAbsorbedWeightedTotal
  ```
  This is the faithful "holes" picture, it **isolates the `F_rad` effect cleanly** (the
  density structure, `shell_mass`, `n_IF` are untouched), and it is a 1-line post-integration
  scaling. The review picked A; B is cleaner physics *and* easier to verify in isolation, which
  is exactly the "check the small `F_rad` effect in isolation" the review asks for.

**Recommendation: Model B**, scoped to the dust/bolometric channel (`f_absorbed`). Keep the
ionising escape `f_esc_ion` and `P_HII` untouched (Phase G owns LyC). If Joel prefers minimal
diff over isolation, Model A is the fallback (then §5.A applies).

## 4. Decisions for Joel (do not silently choose)

- **D1 — model:** B (absorbed-fraction split, recommended) vs A (scale `dτ/dr`, matches scaffold).
- **D2 — coupling source:** constant `coverFraction` now, **or** the breakout-triggered
  `Cf_eff(R2)` once STEP 3 lands. P3 should read whichever via a single helper so the swap is
  one line. (Soft dependency on P2/STEP 3; P3 works with the constant today.)
- **D3 — ionising channel:** keep LyC escape out of P3 (Phase G) — recommended — **or** also
  scale `f_esc_ion_eff = Cf·f_esc_ion + (1−Cf)`. The latter changes `n_IF_Str`/`P_HII` (a
  *dynamical* shell-driving term, `shell_structure.py:234`), so it is a bigger commit and
  overlaps CLOUDY `fleak`; defer.
- **D4 — guard/range:** validate the source `Cf ∈ (0,1]` (already done for `coverFraction`);
  `f_cover = 0` (fully open) must not divide-by-zero anywhere (it doesn't here, but assert).

## 5. Implementation steps (recommended: Model B)

**Step 1 — single coupling helper (where the value comes from).**
In `shell_structure.py`, replace `f_cover = 1` with the leak geometry:
```python
# Photon covering fraction = leak geometry (same holes that vent hot gas).
# Constant coverFraction today; swap to get_effective_coverFraction(R2,...) when
# the breakout trigger (STEP 3) exists. Cf=1 -> f_cover=1 -> no change.
f_cover = params['coverFraction'].value
```
Keep passing `f_cover` into both `get_shellODE` call sites unchanged (so Model-A users can
flip later); for Model B the ODE will be called with `f_cover` but the ODE **ignores** it
(see Step 2), and the scaling happens in Step 3.

**Step 2 — make the ODE covering-agnostic (Model B) and fix the latent asymmetry.**
In `get_shellODE.py`, **remove** the `* f_cover` from the ionised branch (`:100`) so both
branches read `dtaudr = nShell * sigma_dust`. This (a) keeps the structure integration at full
physics and (b) eliminates the ionised/neutral inconsistency by making both branches identical.
(Leave the `f_cover` parameter in the signature to avoid churn, or drop it — Joel's call.)
*If Model A is chosen instead:* do the opposite — **add** `* f_cover` to the neutral branch
(`:122`) so both branches scale, and skip Step 3.

**Step 3 — apply the covering fraction to the absorbed fraction (Model B).**
In `shell_structure.py`, right where `f_absorbed` is formed (`:391`), scale the dust/bolometric
absorbed fraction by the covered fraction before it is returned:
```python
f_absorbed = (f_absorbed_ion * Li + f_absorbed_neu * Ln) / (Li + Ln)
f_absorbed = f_cover * f_absorbed   # (1-Cf) of sightlines are holes -> absorb nothing
```
This flows into `shell_fAbsorbedWeightedTotal` → `F_rad` automatically. Do **not** scale
`f_absorbed_ion`/`f_esc_ion` here (D3: LyC stays with Phase G). Add a comment noting the
dissolved-shell branch (`:407-410`) already sets `f_absorbed = 0`, so the scaling is a no-op
there (correct).

**Step 4 — registered output (audit/plot).** Optionally register `shell_fCover` (the value
used) as a runtime output so the coupling is auditable, mirroring `bubble_Leak`.

## 6. The deliberate inconsistency to document

After P3 (Model B, D3=defer), **dust radiation sees holes but ionising radiation does not**
within the same run. That is intentional: the in-run LyC structure (`n_IF`, `P_HII`) is kept on
the sealed-shell assumption and the LyC leak is accounted once, at the CLOUDY post-processing
stage (Phase G, `fleak = 1−Cf`). State this in a code comment and in the manuscript so it is a
*declared* modelling choice, not an oversight. (Revisit under D3 if the LyC-driven `P_HII` term
turns out to matter for the Rosette — the review expects it to be small.)

## 7. Verification gates (run in order)

1. **Regression / merge gate.** `coverFraction = 1.0` ⇒ `f_cover = 1` ⇒ `f_absorbed`,
   `F_rad`, `R2(t)`, `Eb(t)`, shell structure **bit-identical** to baseline. (Model B: Step 3 is
   `×1`; Step 2 removes a `×1`.)
2. **No-op of Step 2 alone.** With `f_cover = 1`, removing `*f_cover` from `:100` changes
   nothing (it was `×1`) — confirms Step 2 is behaviour-neutral at the baseline.
3. **Monotone `F_rad`.** `coverFraction ∈ {1.0, 0.9, 0.5}` ⇒ smaller `shell_fAbsorbedWeightedTotal`
   and smaller `F_rad` at fixed state; strictly monotone, `f_absorbed → 0` as `Cf → 0`.
4. **Isolation.** Confirm (Model B) that `shell_mass`, `tau_kappa_IR`, `n_IF`, `P_HII` are
   **unchanged** by `Cf < 1` — only `f_absorbed`/`F_rad` move. (This is the advantage over Model A;
   if they move, Step 2/3 leaked into the structure.)
5. **Magnitude check.** Quantify Δ`R2(t)` from `F_rad` reduction at `Cf = 0.9`; the review
   expects it small for the Rosette. Report it.
6. **Coupling swap (when STEP 3 exists).** Switch the Step-1 source to `Cf_eff(R2)`; confirm
   `f_cover = 1` while `R2 < R_breakout`, then ramps — `f_absorbed` follows.

## 8. Do-nots

- Do **not** also scale `dphidr`/`f_esc_ion`/`P_HII` in this commit (that is D3 / Phase G).
- Do **not** leave the ionised/neutral `dtaudr` asymmetric — Step 2 makes them identical (Model B)
  or both-scaled (Model A); never one-sided.
- Do **not** couple `f_cover` to a *frozen* `R2` if using `Cf_eff` — read the live segment `R2`.
- Do **not** double-charge the LyC leak: if Phase G later debits `Q0` by `fleak`, P3 must not
  also be scaling `f_esc_ion`.
- Do **not** treat this `f_cover` as the CLOUDY `fleak`; they share geometry but act in different
  stages (in-run `F_rad` vs post-processing photon budget).
