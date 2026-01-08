# TRINITY Codebase Overview

## Executive Summary

**TRINITY** is a Python-based astrophysics simulation code for modeling the expansion of wind-blown bubbles and superbubbles in molecular clouds. It simulates the evolution of stellar feedback-driven shells through multiple physical phases: energy-driven, transition, and momentum-driven phases.

The code is a continuation/rewrite of the WARPFIELD simulation code, designed to model how massive star clusters drive the expansion of ionized gas shells through their stellar winds and radiation.

---

## Purpose and Scientific Goals

TRINITY models the following physical processes:

1. **Energy-Driven Phase (Phase 1)**:
   - Early expansion driven by stellar wind energy deposition
   - Bubble structure with hot interior and cooler shell
   - Thermal conduction between hot bubble and cold shell

2. **Transition Phase (Phase 1c)**:
   - Transition from energy-driven to momentum-driven expansion
   - Cooling becomes dominant

3. **Momentum-Driven Phase (Phase 2)**:
   - Late-time expansion driven by momentum injection
   - Shell fragmentation and dissolution

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.x |
| **Numerical Computing** | NumPy |
| **ODE Integration** | SciPy (scipy.integrate, scipy.optimize) |
| **Data Format** | JSON/JSONL (line-delimited JSON) |
| **Documentation** | Sphinx with ReadTheDocs |
| **Version Control** | Git |

### Key Dependencies

```
numpy
scipy
yaml
```

---

## Main Entry Points

### 1. Primary Entry Point: `run.py`

```bash
python3 ./run.py param/example.param
```

This script:
- Parses command-line arguments for parameter file path
- Reads parameters via `src/_input/read_param.py`
- Initializes the simulation via `src/main.py`
- Displays header information
- Calls `main.start_expansion(params)`

### 2. Core Simulation Logic: `src/main.py`

The `start_expansion()` function orchestrates the simulation:
1. Initializes cloud properties (`get_InitCloudProp`)
2. Loads Starburst99 stellar feedback data (`read_SB99`)
3. Sets up cooling curves (CIE and non-CIE)
4. Runs the expansion simulation through multiple phases

---

## Directory Structure

```
trinity/
├── run.py                  # Main entry point
├── generate_params.py      # Parameter file generator
├── param/                  # Parameter files (.param)
│   ├── default.param       # Default parameters with documentation
│   └── *.param             # Various simulation configurations
├── src/                    # Source code
│   ├── main.py             # Core simulation orchestration
│   ├── _input/             # Input handling
│   ├── _output/            # Output formatting
│   ├── _functions/         # Utility functions
│   ├── _plots/             # Visualization
│   ├── phase0_init/        # Initialization
│   ├── phase1_energy/      # Energy-driven phase
│   ├── phase1b_energy_implicit/  # Implicit energy phase
│   ├── phase1c_transition/ # Transition phase
│   ├── phase2_momentum/    # Momentum-driven phase
│   ├── bubble_structure/   # Bubble physics
│   ├── shell_structure/    # Shell physics
│   ├── cloud_properties/   # Cloud density profiles
│   ├── cooling/            # Cooling curves (CIE/non-CIE)
│   └── sb99/               # Starburst99 interface
├── lib/                    # External data libraries
│   ├── cooling/            # Cooling curve data
│   └── sps/                # Stellar population synthesis data
├── analysis/               # Refactored/analyzed code versions
├── docs/                   # Sphinx documentation
├── outputs/                # Simulation output directory
└── test/                   # Test scripts
```

---

## Parameter System

TRINITY uses a hierarchical parameter system:

1. **`param/default.param`**: Contains all parameters with:
   - `# INFO:` - Human-readable description
   - `# UNIT:` - Physical units (e.g., `[Msun]`, `[pc]`)
   - Default values

2. **User `.param` files**: Override specific defaults
   - Simple `key value` format
   - Inline comments supported (`# comment`)

3. **`DescribedDict`**: Runtime container storing:
   - `value`: The parameter value (converted to astronomy units)
   - `info`: Description string
   - `ori_units`: Original unit specification

### Example Parameter File

```
mCloud    1e6        # Cloud mass [Msun]
sfe       0.01       # Star formation efficiency
ZCloud    1          # Metallicity [Zsun]
dens_profile  densPL # Density profile type
nCore     1e5        # Core density [cm^-3]
```

---

## Output System

TRINITY uses a JSONL-based snapshot system:

- **`dictionary.jsonl`**: One JSON object per line, each representing a simulation snapshot
- **Append-only writes**: O(1) performance for saving snapshots
- **Array compression**: Long arrays are simplified before saving

### Loading Results

```python
from src._input.dictionary import DescribedDict

# Load all snapshots
snapshots = DescribedDict.load_snapshots('/path/to/output')

# Load specific snapshot
params = DescribedDict.load_snapshot('/path/to/output', snap_id=10)
R2 = params['R2'].value  # Shell radius
t = params['t_now'].value  # Time
```

---

## Physics Modules

### 1. Starburst99 Interface (`src/sb99/`)
- Reads stellar population synthesis data
- Provides interpolated feedback: Lbol, LWind, Qi (ionizing photons), etc.

### 2. Cooling Module (`src/cooling/`)
- **CIE (Collisional Ionization Equilibrium)**: T > 10^5.5 K
- **Non-CIE**: T < 10^5.5 K, uses CLOUDY/OPIATE tables

### 3. Shell Structure (`src/shell_structure/`)
- Solves ODEs for shell density profile
- Calculates ionization fraction, optical depth

### 4. Bubble Structure (`src/bubble_structure/`)
- Calculates bubble luminosity and cooling
- Temperature and density profiles within the bubble

### 5. Density Profiles (`src/cloud_properties/`)
- Bonnor-Ebert spheres (`densBE`)
- Power-law profiles (`densPL`)

---

## References

- TRINITY documentation: https://trinitysf.readthedocs.io/
- Rahner (2018) PhD thesis - Physics equations
- Weaver et al. (1977), ApJ 218, 377 - Bubble dynamics
- Krumholz et al. (2009), ApJ 693, 216 - Feedback models
