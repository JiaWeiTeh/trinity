# Recommendations for TRINITY

This document provides prioritized recommendations for improving TRINITY's architecture, performance, and maintainability.

---

## Priority 1: Critical Fixes (Immediate)

### 1.1 Fix Physics Bugs in Shell Structure

**Issue**: Missing mu factors in shell density equations (identified in `analysis/shell_structure/`)

**Impact**: Shell density calculations wrong by 40-230%

**Solution**: Apply fixes from `analysis/shell_structure/REFACTORED_get_shellODE.py`

```python
# BEFORE (WRONG in src/shell_structure/get_shellODE.py):
dndr = 1/(k_B*t_ion) * (rad_term + recomb_term)

# AFTER (CORRECT):
dndr = (mu_p/mu_n) / (k_B*t_ion) * (rad_term + recomb_term)
```

**Effort**: 1 day
**Impact**: Physics accuracy

---

### 1.2 Replace Manual Euler with scipy.integrate

**Issue**: Manual Euler integration is slow and inaccurate

**Current** (100,000 steps, O(dt) error):
```python
for ii, time in enumerate(t_arr):
    rd, vd, Ed, Td = get_ODE_Edot(y, time, params)
    R2 += rd * dt_min
```

**Recommended** (1,000-10,000 steps, O(dt^4) error):
```python
from scipy.integrate import solve_ivp

sol = solve_ivp(
    fun=ode_function_pure,
    t_span=(t_start, t_end),
    y0=[R2, v2, Eb, T0],
    method='LSODA',  # Adaptive stiff/non-stiff
    rtol=1e-6,
    atol=1e-8
)
```

**Requirements**:
1. Make ODE functions pure (no side effects)
2. Separate state updates from derivative calculations

**Effort**: 3-5 days
**Impact**: 10-100x speedup

---

### 1.3 Replace Grid Search with scipy.optimize

**Issue**: `get_betadelta.py` uses manual 5x5 grid search (25 evaluations)

**Recommended**:
```python
from scipy.optimize import minimize

result = minimize(
    fun=objective_function,
    x0=[beta_guess, delta_guess],
    method='L-BFGS-B',
    bounds=[(0, 1), (-1, 0)]
)
# Converges in ~7 evaluations
```

**Effort**: 1-2 days
**Impact**: 3-4x speedup for optimization loop

---

## Priority 2: Architecture Improvements (Short-term)

### 2.1 Pure Functions for ODE Calculations

**Issue**: ODE functions modify `params` during integration, requiring deepcopy

**Current**:
```python
def get_ODE_Edot(y, t, params):
    params['t_now'].value = t  # Side effect!
    params['R2'].value = y[0]  # Side effect!
    # ...calculations...
    return [dRdt, dvdt, dEdt, dTdt]
```

**Recommended**:
```python
def get_ODE_Edot_pure(y, t, params_readonly):
    """Pure function: only reads params, never writes."""
    R2, v2, Eb, T0 = y
    # Read from params (immutable)
    gamma = params_readonly['gamma_adia'].value
    # Calculate derivatives
    return [dRdt, dvdt, dEdt, dTdt]

# Update params AFTER integration completes
params['R2'].value = sol.y[0, -1]
params['t_now'].value = sol.t[-1]
```

**Effort**: 5 days
**Impact**: Enables scipy.integrate, easier debugging, parallelization-ready

---

### 2.2 Clean Up Legacy Code

**Issue**: Many files with `_old.py`, `_legacy.py`, `_before*.py` suffixes

**Files to archive or remove**:
```
src/_input/read_param_old.py
src/_input/read_param_legacy.py
src/_input/read_param_newer_beforechange.py
src/cooling/CIE/read_coolingcurve_old.py
src/cooling/non_CIE/read_cloudy_old.py
src/phase1_energy/run_energy_phase_old.py
src/phase1_energy/run_energy_phase_before_T0.py
src/phase1_energy/run_energy_phase_beforeClean.py
src/phase1_energy/run_energy_phase_beforeCleanAfterMergephaseODE.py
src/phase1_energy/run_energy_phase_beforeMergephaseODE.py
src/phase1_energy/run_energy_phase_before_Tnowchanges.py
src/bubble_structure/bubble_luminosity_legacy.py
src/bubble_structure/bubble_luminosity_beforeT0.py
src/shell_structure/shell_structure_beforeClean.py
src/sb99/getSB99_data_original.py
src/sb99/read_SB99_old.py
old/  # Entire directory
```

**Recommended action**:
1. Create `archive/` directory
2. Move all legacy files there
3. Update any remaining imports

**Effort**: 1 day
**Impact**: Cleaner codebase, easier navigation

---

### 2.3 Introduce Logging

**Issue**: All output via `print()` statements

**Current**:
```python
print('..loaded sps files.')
print(f'Inner discontinuity: {R1} pc')
print('saving snapshot')
```

**Recommended**:
```python
import logging
logger = logging.getLogger(__name__)

logger.info('Loaded SPS files')
logger.debug(f'Inner discontinuity R1={R1:.4f} pc')
logger.warning('Shell approaching fragmentation threshold')
```

**Setup**:
```python
# In run.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('trinity.log'),
        logging.StreamHandler()
    ]
)
```

**Effort**: 2 days
**Impact**: Better debugging, controllable verbosity

---

## Priority 3: Structural Reorganization (Medium-term)

### 3.1 Reorganize Module Structure

**Current**: Flat structure with inconsistent naming

**Recommended structure**:
```
src/
├── __init__.py
├── core/                       # Core simulation engine
│   ├── __init__.py
│   ├── simulation.py           # Main Simulation class
│   ├── state.py                # SimulationState dataclass
│   └── integrator.py           # ODE integration wrapper
│
├── config/                     # Configuration handling
│   ├── __init__.py
│   ├── parameters.py           # DescribedDict, DescribedItem
│   ├── reader.py               # Parameter file parsing
│   └── defaults.py             # Default values
│
├── physics/                    # Physics modules
│   ├── __init__.py
│   ├── bubble/
│   │   ├── __init__.py
│   │   ├── structure.py
│   │   └── luminosity.py
│   ├── shell/
│   │   ├── __init__.py
│   │   ├── structure.py
│   │   └── ode.py
│   ├── cooling/
│   │   ├── __init__.py
│   │   ├── cie.py
│   │   └── non_cie.py
│   └── cloud/
│       ├── __init__.py
│       ├── density_profiles.py
│       └── mass_profile.py
│
├── feedback/                   # Stellar feedback
│   ├── __init__.py
│   └── starburst99.py
│
├── phases/                     # Simulation phases
│   ├── __init__.py
│   ├── base.py                 # Abstract Phase class
│   ├── energy.py               # Phase 1
│   ├── implicit.py             # Phase 1b
│   ├── transition.py           # Phase 1c
│   └── momentum.py             # Phase 2
│
├── io/                         # Input/Output
│   ├── __init__.py
│   ├── writer.py               # Output writers (JSONL, HDF5)
│   ├── reader.py               # Snapshot loaders
│   └── history.py              # Time series output
│
└── utils/                      # Utilities
    ├── __init__.py
    ├── units.py                # Unit conversions
    ├── constants.py            # Physical constants
    └── math.py                 # Mathematical utilities
```

**Effort**: 1-2 weeks
**Impact**: Better maintainability, clearer dependencies

---

### 3.2 Introduce Abstract Phase Class

**Recommended**:
```python
# src/phases/base.py
from abc import ABC, abstractmethod

class Phase(ABC):
    """Abstract base class for simulation phases."""

    def __init__(self, params, config):
        self.params = params
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def initialize(self):
        """Set up initial conditions for this phase."""
        pass

    @abstractmethod
    def evolve(self, t_start, t_end):
        """Evolve the system from t_start to t_end."""
        pass

    @abstractmethod
    def check_termination(self) -> bool:
        """Check if phase should terminate."""
        pass

    def run(self):
        """Main entry point for running the phase."""
        self.initialize()
        while not self.check_termination():
            self.evolve(self.t_current, self.t_next)
            self.save_snapshot()
```

**Effort**: 3-5 days
**Impact**: Consistent phase interface, easier to add new phases

---

### 3.3 Implement HDF5 Output

**Recommended**:
```python
# src/io/writer.py
import h5py
from datetime import datetime

class HDF5Writer:
    def __init__(self, filepath, params):
        self.filepath = filepath
        self.params = params
        self._init_file()

    def _init_file(self):
        with h5py.File(self.filepath, 'w') as f:
            # Metadata
            f.attrs['created'] = str(datetime.now())
            f.attrs['trinity_version'] = '2.0'

            # Parameter group
            config = f.create_group('config')
            for key, item in self.params.items():
                if isinstance(item.value, (int, float, str, bool)):
                    config.attrs[key] = item.value

            # Time series (extensible)
            ts = f.create_group('time_series')
            ts.create_dataset('t', (0,), maxshape=(None,), dtype='f8')
            ts.create_dataset('R2', (0,), maxshape=(None,), dtype='f8')
            # ... more datasets

    def append_snapshot(self, snapshot):
        with h5py.File(self.filepath, 'a') as f:
            ts = f['time_series']
            n = ts['t'].shape[0]
            for key in ['t', 'R2', 'v2', 'Eb', 'T0']:
                ts[key].resize(n + 1, axis=0)
                ts[key][n] = snapshot[key]
```

**Effort**: 2-3 days
**Impact**: Smaller files, faster I/O, better interoperability

---

## Priority 4: Testing and Validation (Medium-term)

### 4.1 Add Unit Tests

**Recommended structure**:
```
tests/
├── conftest.py              # Shared fixtures
├── test_config/
│   ├── test_reader.py
│   └── test_parameters.py
├── test_physics/
│   ├── test_bubble.py
│   ├── test_shell.py
│   └── test_cooling.py
├── test_phases/
│   ├── test_energy.py
│   └── test_momentum.py
└── test_integration/
    └── test_full_simulation.py
```

**Example test**:
```python
# tests/test_physics/test_bubble.py
import pytest
import numpy as np
from src.physics.bubble import structure

@pytest.fixture
def sample_params():
    """Create sample parameters for testing."""
    from src.config.parameters import DescribedDict, DescribedItem
    params = DescribedDict()
    params['gamma_adia'] = DescribedItem(5/3)
    params['R2'] = DescribedItem(1.0)  # pc
    params['R1'] = DescribedItem(0.1)  # pc
    params['Eb'] = DescribedItem(1e4)  # energy units
    return params

def test_pressure_positive(sample_params):
    """Bubble pressure should always be positive."""
    P = structure.compute_pressure(sample_params)
    assert P > 0

def test_pressure_scales_with_energy(sample_params):
    """Pressure should scale linearly with energy."""
    P1 = structure.compute_pressure(sample_params)
    sample_params['Eb'].value *= 2
    P2 = structure.compute_pressure(sample_params)
    assert np.isclose(P2, 2 * P1, rtol=1e-6)
```

**Effort**: 1-2 weeks
**Impact**: Catch regressions, document expected behavior

---

### 4.2 Add Validation Against Analytical Solutions

**Weaver Solution (Energy-Conserving Bubble)**:
```python
def weaver_solution(t, L_wind, rho_0):
    """
    Analytical solution for energy-conserving bubble.
    Weaver et al. (1977), ApJ 218, 377, Eq. 18
    """
    alpha = 0.88
    R = alpha * (L_wind / rho_0)**(1/5) * t**(3/5)
    v = (3/5) * R / t
    return R, v

def test_energy_phase_matches_weaver():
    """Verify energy phase matches Weaver solution."""
    params = create_homogeneous_cloud_params()
    run_energy_phase(params)

    R_numerical = params['R2'].value
    R_analytical, _ = weaver_solution(
        params['t_now'].value,
        params['LWind'].value,
        params['rho_0'].value
    )

    # Should match within 10% for homogeneous cloud
    assert abs(R_numerical - R_analytical) / R_analytical < 0.10
```

**Effort**: 3-5 days
**Impact**: Physics validation

---

## Priority 5: Future Enhancements (Long-term)

### 5.1 Parallelization

**Options**:
1. **Parameter studies**: `multiprocessing.Pool` for independent runs
2. **Single simulation**: Currently not parallelizable due to serial physics

**Quick win for parameter studies**:
```python
from multiprocessing import Pool

def run_single_simulation(param_file):
    params = read_param(param_file)
    main.start_expansion(params)
    return params['path2output'].value

with Pool(processes=4) as pool:
    results = pool.map(run_single_simulation, param_files)
```

---

### 5.2 Configuration Validation with Pydantic

```python
from pydantic import BaseModel, Field, validator

class CloudConfig(BaseModel):
    mass: float = Field(..., gt=0, description="Cloud mass [Msun]")
    metallicity: float = Field(1.0, ge=0, le=10)
    density_profile: str = Field("power_law")

    @validator('density_profile')
    def valid_profile(cls, v):
        if v not in ['power_law', 'bonnor_ebert']:
            raise ValueError(f"Invalid density profile: {v}")
        return v

class SimulationConfig(BaseModel):
    cloud: CloudConfig
    output_dir: str = "./outputs"
    verbose: int = Field(1, ge=0, le=3)
```

---

### 5.3 Restart Capability

```python
class Checkpoint:
    """Save and restore complete simulation state."""

    @staticmethod
    def save(params, filepath):
        state = {
            'params': {k: v.value for k, v in params.items()},
            'arrays': {k: v.value.tolist() for k, v in params.items()
                      if isinstance(v.value, np.ndarray)},
            'phase': params['current_phase'].value,
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'r') as f:
            state = json.load(f)
        params = DescribedDict()
        # Reconstruct params...
        return params
```

---

## Summary: Implementation Roadmap

| Phase | Tasks | Effort | Impact |
|-------|-------|--------|--------|
| **Week 1-2** | Fix physics bugs, replace Euler with scipy | 5-7 days | Critical |
| **Week 3** | Clean up legacy code, add logging | 3 days | High |
| **Week 4-6** | Reorganize modules, pure functions | 2 weeks | High |
| **Week 7-8** | Add unit tests, validation | 2 weeks | Medium |
| **Future** | HDF5 output, parallelization, restart | Ongoing | Medium |

---

## Quick Wins (< 1 day each)

1. Delete unused `_old.py` files
2. Add `__all__` to `__init__.py` files
3. Replace `print()` with `logging.info()`
4. Add type hints to key functions
5. Create `requirements.txt` for pip installation
