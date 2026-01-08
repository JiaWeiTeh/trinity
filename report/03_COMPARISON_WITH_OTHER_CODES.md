# Comparison with Other Astrophysics Codes

This document compares TRINITY's architecture with established astrophysics simulation codes: **MESA**, **FLASH**, and **Athena++**.

---

## Overview Comparison Table

| Aspect | TRINITY | MESA | FLASH | Athena++ |
|--------|---------|------|-------|----------|
| **Language** | Python | Fortran | Fortran/C | C++ |
| **Domain** | 1D bubble expansion | 1D stellar evolution | 3D hydro + nuclear | 3D MHD |
| **Parallelism** | None (serial) | OpenMP | MPI + AMR | MPI + AMR |
| **Integration** | Manual Euler | CVODE (adaptive) | PPM/MUSCL | RK4/VL2 |
| **Config Format** | Custom .param | Fortran namelists | Custom .par | Custom .athinput |
| **Output Format** | JSONL | Custom binary + HDF5 | HDF5 | HDF5/VTK |
| **Modularity** | Medium | High | Very High | Very High |
| **Testing** | Minimal | Extensive | Extensive | Extensive |

---

## Detailed Comparison

### 1. Module Organization

#### TRINITY (Current)

```
src/
├── phase1_energy/      # Phase-specific
├── phase2_momentum/    # Phase-specific
├── bubble_structure/   # Physics
├── shell_structure/    # Physics
└── cooling/            # Physics
```

**Pattern**: Flat structure with phase-specific and physics directories mixed.

#### MESA (Best Practice)

```
star/
├── public/             # Public API (what users import)
├── private/            # Internal implementation
├── test/               # Unit tests
├── make/               # Build system
└── defaults/           # Default configurations

eos/                    # Separate physics modules
├── public/
├── private/
└── test/

kap/                    # Opacity module (same structure)
net/                    # Nuclear networks (same structure)
```

**Pattern**: Each physics module is completely self-contained with:
- `public/`: API definitions
- `private/`: Implementation details
- `test/`: Module-specific tests
- Clear versioning and defaults

#### Athena++ (Best Practice)

```
src/
├── bvals/              # Boundary conditions
├── coordinates/        # Coordinate systems
├── eos/                # Equation of state
├── hydro/              # Hydrodynamics
├── mesh/               # Grid management
├── outputs/            # Output writers
├── pgen/               # Problem generators
└── utils/              # Utilities
```

**Pattern**: Physics and infrastructure clearly separated. Each module defines:
- `*_tasks.cpp`: Task-based parallelism hooks
- `*_diffusion.cpp`: Diffusion operators
- `*_flux.cpp`: Flux calculations

---

### 2. Configuration Systems

#### TRINITY

```
# Simple key-value format
mCloud    1e6
sfe       0.01
dens_profile    densPL
```

**Pros**: Simple, human-readable
**Cons**: No validation, no schema, no type hints

#### MESA (Fortran Namelists)

```fortran
&star_job
  load_saved_model = .false.
  pgstar_flag = .false.
/ ! end star_job

&controls
  initial_mass = 1.0
  initial_z = 0.02
/ ! end controls
```

**Pros**: Type-checked by Fortran, grouped by category
**Cons**: Fortran-specific syntax

#### FLASH (Parameter Files)

```
# Driver parameters
dtinit = 1.0E-15
dtmax  = 1.0E-5
nend   = 10000000

# Grid parameters
geometry = "cartesian"
xmin     = 0.
xmax     = 1.
```

**Pattern**: Clear grouping with comments, extensive defaults system.

#### Recommended for TRINITY

```yaml
# YAML-based configuration (proposed)
simulation:
  name: "my_simulation"
  output_dir: "./outputs"

cloud:
  mass: 1.0e6  # Msun
  metallicity: 1.0  # Zsun
  density_profile:
    type: "power_law"  # or "bonnor_ebert"
    alpha: 0.0

physics:
  cooling:
    cie_table: "default"
    non_cie: true
```

---

### 3. Time Integration

#### TRINITY (Current)

```python
# Manual Euler integration
for ii, time in enumerate(t_arr):
    rd, vd, Ed, Td = get_ODE_Edot(y, time, params)
    R2 += rd * dt_min
    v2 += vd * dt_min
```

**Issues**:
- First-order accuracy (O(dt) error)
- Fixed timestep (inefficient)
- ~100,000 steps for 0.003 Myr

#### MESA (CVODE)

Uses LLNL's CVODE solver:
- Adaptive timestep (controlled by tolerance)
- Variable-order BDF methods (up to 5th order)
- Newton iteration for implicit steps
- ~1,000-10,000 steps for entire evolution

#### Athena++ (Runge-Kutta)

- 2nd or 4th order Runge-Kutta
- CFL-limited timestep (adapts to physics)
- Subcycling for different physics modules

#### Recommended for TRINITY

```python
from scipy.integrate import solve_ivp

sol = solve_ivp(
    fun=lambda t, y: ode_function(t, y, params),
    t_span=(t_start, t_end),
    y0=initial_state,
    method='LSODA',  # Adaptive stiff/non-stiff
    events=[shell_dissolved, cloud_escaped],  # Event detection
    dense_output=True,
    rtol=1e-6,
    atol=1e-8
)
```

---

### 4. Output and Restart Capabilities

#### TRINITY

```python
# JSONL output (one line per snapshot)
{"t_now": 0.001, "R2": 0.5, "v2": 100, ...}
{"t_now": 0.002, "R2": 0.6, "v2": 95, ...}
```

**Pros**: Human-readable, append-only (O(1) writes)
**Cons**: No compression, no restart capability, redundant data

#### MESA (HDF5 + Binary)

- Primary evolution in binary `.mod` files
- HDF5 for post-processing
- Complete restart capability
- `history.data`: Time series in ASCII
- `profile*.data`: Radial profiles

#### FLASH (HDF5 + Checkpoints)

- HDF5 for grid data
- Parallel I/O with MPI-IO
- Checkpoint files for exact restart
- Plot files for analysis

#### Recommended for TRINITY

1. **Primary output**: HDF5 with compression
2. **Restart files**: Complete state dump every N snapshots
3. **History file**: Time series in CSV/ASCII for quick analysis

```python
import h5py

with h5py.File('output.h5', 'w') as f:
    time_series = f.create_group('time_series')
    time_series.create_dataset('t', data=t_arr, compression='gzip')
    time_series.create_dataset('R2', data=R2_arr, compression='gzip')

    profiles = f.create_group('profiles')
    for i, snap in enumerate(snapshots):
        g = profiles.create_group(f'snap_{i:04d}')
        g.create_dataset('bubble_r', data=snap.bubble_r)
        g.create_dataset('bubble_T', data=snap.bubble_T)
```

---

### 5. Testing and Validation

#### TRINITY

- Minimal testing (`test/` directory with a few scripts)
- No unit tests
- No continuous integration

#### MESA

- Extensive test suite (`test_suite/`)
- Automated nightly testing
- Known test cases with expected results
- Performance regression testing

#### Athena++

- Unit tests for each module
- Regression tests with gold standards
- Problem generators for validation

#### Recommended for TRINITY

```python
# tests/test_bubble_structure.py
import pytest
from src.bubble_structure import bubble_luminosity

def test_weaver_solution():
    """Compare with Weaver et al. (1977) analytical solution"""
    params = create_test_params()
    result = bubble_luminosity.get_bubbleproperties(params)

    # Compare with analytical solution
    expected_R = weaver_R(params['t_now'].value)
    assert abs(result['R2'] - expected_R) / expected_R < 0.01

def test_energy_conservation():
    """Verify energy is conserved within tolerance"""
    E_initial = compute_total_energy(params_initial)
    E_final = compute_total_energy(params_final)
    E_input = compute_input_energy(params)
    E_lost = compute_cooling_losses(params)

    assert abs(E_final - E_initial - E_input + E_lost) < 1e-6
```

---

### 6. Documentation

#### TRINITY

- Basic README.md
- ReadTheDocs documentation (external)
- Inline comments

#### MESA

- Extensive online documentation
- Paper references in code
- Interactive tutorials
- Annual summer schools

#### Athena++

- Method papers cited in code
- Wiki documentation
- Problem generator examples

#### Recommended for TRINITY

1. **Docstrings**: NumPy-style with equation references
2. **Method papers**: Cite Rahner thesis, Weaver et al. in code
3. **Examples**: Jupyter notebooks for common use cases
4. **Validation**: Document comparison with analytical solutions

```python
def compute_bubble_pressure(E: float, R2: float, R1: float, gamma: float) -> float:
    """
    Compute bubble pressure from energy.

    Uses the equation from Weaver et al. (1977), ApJ 218, 377, Eq. 12:

        P_b = (gamma - 1) * E_b / (4/3 * pi * (R2^3 - R1^3))

    Parameters
    ----------
    E : float
        Bubble thermal energy [Msun * pc^2 / Myr^2]
    R2 : float
        Outer bubble radius [pc]
    R1 : float
        Inner bubble radius [pc]
    gamma : float
        Adiabatic index (typically 5/3)

    Returns
    -------
    float
        Bubble pressure [Msun / (pc * Myr^2)]

    References
    ----------
    .. [1] Weaver et al. (1977), ApJ 218, 377
    .. [2] Rahner (2018), PhD Thesis, Eq. 2.12
    """
    volume = (4/3) * np.pi * (R2**3 - R1**3)
    return (gamma - 1) * E / volume
```

---

## Key Takeaways

| Best Practice | Source | Applicability to TRINITY |
|---------------|--------|--------------------------|
| Module isolation with public/private split | MESA | High |
| Adaptive time integration | MESA, Athena++ | Critical |
| HDF5 output with restart capability | FLASH, MESA | Medium |
| Task-based parallelism | Athena++ | Future work |
| Extensive test suite | All | High |
| Physics validation against analytical solutions | All | High |

---

## References

- [MESA Documentation](https://docs.mesastar.org/)
- [Athena++ GitHub](https://www.athena-astro.app/)
- [FLASH Code](https://flash.rochester.edu/)
- [Awesome Astrophysical Simulation Codes](https://github.com/pmocz/awesome-astrophysical-simulation-codes)
