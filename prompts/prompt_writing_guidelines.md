# TRINITY Prompt Writing Guidelines

Lessons learned from prompt review sessions. Follow these rules when writing
implementation prompts for Claude or other LLM assistants working on TRINITY.

## 1. File paths must be exact

TRINITY uses per-phase subdirectories, NOT a flat `src/phase_runners/` folder.

Correct paths:
```
src/phase1_energy/run_energy_phase_modified.py
src/phase1_energy/energy_phase_ODEs_modified.py
src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py
src/phase1b_energy_implicit/get_betadelta_modified.py
src/phase1c_transition/run_transition_phase_modified.py
src/phase2_momentum/run_momentum_phase_modified.py
```

**Rule:** Always verify file paths against the repo before including them in a
prompt. An LLM given wrong paths will waste time searching or hallucinate code.

## 2. Units: everything in main() is code units

All numeric params are auto-converted from CGS to code units `[Msun, pc, Myr]`
by `read_param.py` (lines ~248-249) via `cvt.convert2au(unit)`. Once inside
`main()`, no value is in CGS.

| Quantity | CGS (input) | Code unit (in params) |
|----------|-------------|----------------------|
| Number density | cm^-3 | 1/pc^3 (factor: `ndens_cgs2au = 2.938e+55`) |
| Energy | erg | Msun*pc^2/Myr^2 |
| Pressure | dyne/cm^2 | Msun/Myr^2/pc |
| Velocity | cm/s | pc/Myr |
| Boltzmann k_B | erg/K | Msun*pc^2/Myr^2/K |
| Mass | g | Msun |
| Length | cm | pc |
| Time | s | Myr |
| Recomb. coeff. | cm^3/s | pc^3/Myr |

**Rule:** Never write `[cm^-3]` or `[erg]` in prompts when describing values
read from `params['X'].value`. Always use code units. Only plotting scripts
convert back to CGS.

**Evidence:** `main.py` does `params['nCore'].value * cvt.ndens_au2cgs` to
convert *back* to cm^-3 for logging, proving stored values are code units.

## 3. Param access patterns

Physical constants and parameters are accessed differently depending on
where they are declared:

| Constant | Where declared | How to access |
|----------|---------------|---------------|
| `caseB_alpha` | `param/default.param` (unit: `cm**3*s**-1`) | `params['caseB_alpha'].value` (code units) |
| `k_B` | `param/default.param` (unit: `erg*K**-1`) | `params['k_B'].value` (code units) |
| `TShell_ion` | `param/default.param` (unit: `K`) | `params['TShell_ion'].value` (K, no conversion) |
| `mu_ion` | `param/default.param` (unit: `m_H`) | `params['mu_ion'].value` (Msun) |
| `Qi` | SB99 feedback interpolation | `feedback.Qi` -> copied to `params['Qi'].value` via `updateDict` |
| `nCore`, `nISM` | `param/default.param` (unit: `cm**-3`) | `params['nCore'].value` (1/pc^3) |

**Rule:** Do NOT say "check `unit_conversions.py` for caseB_alpha" — it's not
there. `unit_conversions.py` contains *conversion factors* (e.g., `ndens_cgs2au`),
not physical constants. Physical constants live in `default.param` and are
auto-converted.

## 4. Describe existing code accurately

When a prompt says "change X to Y", it must describe the current state of X
correctly. Common pitfalls:

- **Phase branching in ODEs:** `get_ODE_Edot_pure()` already branches on
  `snapshot.current_phase == 'transition'` to add P_ram. Don't describe
  it as a simple `P_drive = max(Pb, P_HII)` — the transition logic exists.

- **Variable access per phase:** R2, Pb, Qi are obtained differently in each
  phase runner. Don't assume a uniform access pattern — read each file.

- **Dataclass structures:** If code uses an `ODESnapshot` dataclass, the
  prompt must mention `create_ODE_snapshot()` as the place to populate it.

**Rule:** Before writing a prompt, read the actual code at the modification
points. Never describe code from memory or assumption.

## 5. Prompt structure checklist

A good TRINITY implementation prompt should have:

- [ ] **Physics summary** — what and why, with key formulae
- [ ] **READ FIRST section** — files to understand before coding, with
      specific things to look for in each file
- [ ] **Implementation plan** — ordered steps with code snippets
- [ ] **DO NOT change list** — explicit guardrails for things to preserve
- [ ] **Edge cases** — physically motivated boundary conditions
- [ ] **Verification checklist** — concrete testable assertions

## 6. Common mistakes to avoid

1. **Wrong unit labels in docstrings** — historically, many TRINITY docstrings
   incorrectly claimed CGS units. When referencing existing docstrings, verify
   against `read_param.py`'s auto-conversion, not the docstring itself.

2. **Assuming flat directory structure** — TRINITY organises phase runners
   into `src/phase1_energy/`, `src/phase1b_energy_implicit/`, etc.

3. **Conflating conversion constants with physical constants** —
   `unit_conversions.py` has `k_B_cgs2au` (a conversion factor), but `k_B`
   the physical constant is a param in `default.param`.

4. **Forgetting the params access pattern** — values are `params['X'].value`
   (using DescribedItem), not `params['X']` directly.

5. **Not specifying where new params are initialised** — new params fields
   must be added in `src/_input/read_param.py` alongside existing runtime
   params like `P_HII`, `P_drive`, `P_ram`.
