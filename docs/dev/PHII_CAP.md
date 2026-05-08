# P_HII / P_b Cap Issue

**Status:** open — design choice with consequences for the `include_PHII` study.
**Branch:** `claude/density-profile-blend-tests-OrFsV`
**Discovered:** investigating why `include_PHII=True/False` produce nearly
identical `R(t)` for `1e6_sfe010_n1e3_PL0`.

## TL;DR

In the energy/implicit phases, `include_PHII` is **inert by construction**.
Two model choices act together:

1. A cap inside `shell_structure_modified.py` pins
   `n_IF_Str ≤ shell_n0`, which forces
   `P_HII ≤ 2*(mu_ion/mu_atom) * P_b ≈ 0.957 * P_b`.
2. The energy/implicit-phase rhs uses
   `P_drive = max(P_b, P_HII)`.

Combined: the `max` gate always selects `P_b`, in any cell where the cap
binds (which is most cells of interest). So yesPHII and noPHII run with
bit-identical rhs through the entire energy phase. The slight late-time
yesPHII–noPHII separation in `|v_2|(t)` comes from the **momentum phase**
(`P_drive = P_HII + P_ram`, additive, no `max`), not from PHII actually
"winning" anywhere.

## Code references

### The cap

`src/shell_structure/shell_structure_modified.py`

- Line 115–117 — define `shell_n0` from `P_b`:
  ```python
  nShell0 = (params['mu_ion'].value / params['mu_atom'].value /
             (params['k_B'].value * params['TShell_ion'].value) * params['Pb'].value)
  shell_n0 = nShell0
  ```
- Line 236–244 — Strömgren density, then the cap:
  ```python
  if (_vol_ion > 0.0) and (_Qi_absorbed > 0.0):
      n_IF_Str = sqrt(3 * _Qi_absorbed / (4*pi * alpha_B * _vol_ion))
      n_IF_Str = min(n_IF_Str, shell_n0)   # ← the cap
  else:
      n_IF_Str = 0.0
  ```

### The drive selectors

| Phase      | File                                                          | Line | Formula                              |
|------------|---------------------------------------------------------------|-----:|--------------------------------------|
| Energy     | `src/phase1_energy/energy_phase_ODEs_modified.py`             |  256 | `P_drive = max(P_b, P_HII)`          |
| Implicit   | `src/phase1_energy/energy_phase_ODEs_modified.py` (shared rhs)|  256 | `P_drive = max(P_b, P_HII)`          |
| Transition | `src/phase1c_transition/run_transition_phase_modified.py`     |  327 | `P_drive = max(P_b, P_HII + P_ram)`  |
| Momentum   | `src/phase2_momentum/run_momentum_phase_modified.py`          |  261 | `P_drive = P_HII + P_ram`            |

### P_HII formula (each phase runner)

```python
P_HII = 2.0 * n_IF_Str * params['k_B'].value * params['TShell_ion'].value
params['P_HII'].value = P_HII
```
Locations: `run_energy_phase_modified.py:195`,
`run_energy_implicit_phase_modified.py:634`,
`run_transition_phase_modified.py:524`,
`run_momentum_phase_modified.py:592`.

### The cap ratio

With `mu_atom = 14/11` and `mu_ion = 14/23`
(`src/_plots/paper_InitialCloudRadius.py:33,35`):

```
P_HII / P_b  =  2 * (mu_ion / mu_atom)  =  22/23  ≈  0.9565
log10(P_HII / P_b)  ≈  -0.019
```

That fixed 4.3% deficit is invisible on a log axis spanning many decades —
which is why `Pb(t)` and `P_HII(t)` overlay so perfectly in
`paper_phii_window` plots.

## Why it shows up most for high-feedback cells

- `n_IF_Str_raw = sqrt(3 * Qi / (4*pi*alpha_B * dV))` is the Strömgren density.
- It binds against `shell_n0 = (mu_ion/mu_atom) * P_b / (k_B T_ion)`.
- Cap binds when `n_IF_Str_raw > shell_n0`, i.e. when ionising photons are
  abundant relative to bubble pressure.
- High `sfe`, high `M_cl`, modest `n_ISM` ⇒ lots of `Qi`, modest `P_b` ⇒ cap
  binds throughout the simulation.
- Low feedback (small cluster, tenuous gas) ⇒ Strömgren density may stay
  below `shell_n0` ⇒ cap not binding ⇒ `P_HII < P_b` genuinely. But
  `max(P_b, P_HII)` still picks `P_b`, so the energy phase is **still**
  unaffected.

The corollary is uncomfortable: the energy-phase `max(P_b, P_HII)` gate is
**unable to ever pick P_HII**, independent of the cap. The cap just makes
the equality nearly tight.

## The diagnostic

`src/_plots/paper_phii_window.py` produces a 3-panel figure per matched
`_yesPHII`/`_noPHII` pair:

- **Top:** `Pb(t)` and `P_HII(t)` (yesPHII run), with the `P_HII > Pb` window
  shaded.
- **Middle:** `log10(P_HII / Pb)`. Reference lines at `0` (parity) and
  `log10(22/23) ≈ -0.019` (cap value). Also reports the percentage of
  samples sitting within 0.05 dex of the cap.
- **Bottom:** `|v_2|(t)` for both runs.

Run with:
```
python src/_plots/paper_phii_window.py -F outputs/<sweep_folder>
python src/_plots/paper_phii_window.py -F outputs/<sweep_folder> -n 1e3 --mCloud 1e6 --sfe 010
```

For `1e6_sfe010_n1e3_PL0`, the middle panel sits flat at `~ -0.019` for
nearly the whole run: the cap is binding everywhere.

## Is the cap "wrong"?

Not exactly — both the cap and the `max` gate are physically defensible
in isolation:

- **Cap.** A photoionised skin in pressure contact with the bubble cannot
  sustain pressure higher than its container under hydrostatic balance;
  excess pressure would expand the skin until equilibrium. This is the
  argument cited in the comment at line 241.
- **`max(P_b, P_HII)` gate.** If both regions push on the cold shell, the
  shell sees the dominant pressure. (Though "dominant" is debatable: under
  some configurations the two should add, see momentum phase.)

But **together** they produce a circular result: the cap says "P_HII can
never exceed P_b"; the gate says "drive picks the larger of P_b, P_HII".
Conclusion: drive equals P_b, always, in the energy phase. Whether
`include_PHII` is `True` or `False` cannot change the trajectory there.

## What to do (options, ordered by invasiveness)

### A. Accept the design, document, scope the paper

Reframe the `include_PHII` study as a **momentum-phase** study: PHII only
contributes additively in phase 2, and the paper's statement should be
"adding PHII to the momentum phase yields a few-percent boost in `R(t)`
in this regime." Cleanest, no code changes.

### B. Replace `max` with addition in the energy/implicit phases

Change `energy_phase_ODEs_modified.py:256`:
```python
P_drive = max(press_bubble, P_HII)
# →
P_drive = press_bubble + P_HII   # or  press_bubble + f_couple * P_HII
```
Justification: pressures from independent regions of gas in contact with
the shell add. The bubble pushes from the inside; the photoionised layer,
sitting between the bubble and the cold shell, also pushes outward on the
cold shell.

Caveats:
- Even after this change, the cap still pins `P_HII ≈ 0.957 P_b`, so the
  effective drive becomes `~1.957 P_b`. That's a uniform ~factor-2 boost
  to drive in cap-binding regimes, propagating into `R(t)`.
- May break agreement with whatever was tuned/calibrated under the
  current rhs. Comparison runs needed.

### C. Use the existing `pressure_blend.py`

`src/phase_general/pressure_blend.py` already implements a continuous,
absorption-weighted blend:
```
w_blend = f_abs_ion * P_HII_Str / (P_HII_Str + P_b)
```
that smoothly interpolates between energy-driven (`w → 0`) and
momentum-driven (`w → f_abs_ion`) regimes. **It is not currently
imported anywhere.** Wire it into the energy/implicit/transition phases
in place of `max(...)`.

This is probably the right long-term fix because the module appears to
have been designed exactly for this: the file's docstring describes the
three-regime behaviour cleanly. The presence of an unused blend utility
suggests there was an intent to migrate but it didn't land.

### D. Loosen or rederive the cap

If you keep the `max` gate, drop the cap (or replace `min(n_IF_Str, shell_n0)`
with a softer barrier). Risks: in transient/non-equilibrium states, the
Strömgren formula may give unphysically large `n_IF_Str` and `P_HII`,
potentially destabilising the rhs. Less recommended than C.

## Recommendation

Pursue **A** (document) for immediate paper deliverables, and queue **C**
(wire up `pressure_blend.py`) as the long-term fix. **C** is the cleanest
because (i) it removes the circularity, (ii) it uses code that already
exists in the repo, and (iii) its three-regime behaviour matches the
physics intuition of when each pressure should dominate.

Before adopting **C**, run a small sweep with the blended rhs to confirm
it doesn't destabilise the LSODA implicit-phase integration that we just
finished smoothing. The blend involves a denominator `P_HII_Str + P_b`
that's well-behaved, so this is more of a sanity check than a real risk.

## Quick verification numbers

For `1e6_sfe010_n1e3_PL0_yesPHII`, the diagnostic plot shows:

- Top panel: `Pb(t)` and `P_HII(t)` indistinguishable on log y across ~11
  decades.
- Middle panel: `log10(P_HII/Pb)` flat at `~-0.019` for essentially the
  whole run. Cap-binding fraction near 100%.
- Bottom panel: yesPHII/noPHII `|v_2|` overlay until `t ≳ 2.3 Myr`
  (momentum phase begins), then yesPHII rises a few percent above noPHII —
  consistent with the additive momentum-phase rhs.

For low-feedback cells (e.g. `sfe=001, n=1e3, M_cl=1e5`) the middle panel
should drop below the cap line for at least part of the run (Strömgren
density falls below `shell_n0`) — but yesPHII/noPHII would still match in
the energy phase because of the `max` gate. The momentum-phase difference
might be slightly larger because `Qi/R^2` carries more weight relative to
ram pressure.
