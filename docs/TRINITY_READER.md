# TRINITY Output Reader

A comprehensive Python module for reading and analyzing TRINITY simulation output files. Similar to `astropy.io.fits`, it provides a clean, Pythonic API for accessing simulation data.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Classes](#core-classes)
  - [TrinityOutput](#trinityoutput)
  - [Snapshot](#snapshot)
- [Reading Data](#reading-data)
  - [Opening Files](#opening-files)
  - [Accessing Time Series](#accessing-time-series)
  - [Getting Snapshots](#getting-snapshots)
  - [Filtering Data](#filtering-data)
- [Interpolation](#interpolation)
- [Path Utilities](#path-utilities)
  - [Finding Single Files](#finding-single-files)
  - [Batch Processing](#batch-processing)
  - [Grid Organization](#grid-organization)
- [Available Parameters](#available-parameters)
- [Examples](#examples)
- [API Reference](#api-reference)

---

## Overview

The `trinity_reader` module is the recommended way to access TRINITY simulation output data. It replaces direct JSON parsing and provides:

- **Easy data access**: Get any parameter as a numpy array with a single call
- **Time-based queries**: Get snapshots at specific times with interpolation support
- **Filtering**: Filter by simulation phase or time range
- **Batch processing**: Find and organize multiple simulations for grid plots
- **Self-documenting**: Built-in parameter documentation via `output.info()`

## Installation

The module is part of the TRINITY package. Import it as:

```python
from src._output.trinity_reader import (
    TrinityOutput,      # Main reader class
    load_output,        # Convenience function (alias for read)
    find_data_file,     # Find data file for a run name
    find_all_simulations,  # Find all simulations in a folder
    organize_simulations_for_grid,  # Organize for grid plots
    get_unique_ndens,   # Get unique density values
)
```

## Quick Start

```python
from src._output.trinity_reader import load_output

# Load a simulation
output = load_output('/path/to/simulation/dictionary.jsonl')

# Get basic info
output.info()

# Access time series data
times = output.get('t_now')      # numpy array of times [Myr]
radii = output.get('R2')         # shell radii [pc]
velocity = output.get('v2')      # expansion velocity [pc/Myr]

# Get snapshot at a specific time
snap = output.get_at_time(1.0)   # interpolated snapshot at t=1 Myr
print(snap['R2'], snap['v2'])

# Filter by phase
energy_phase = output.filter(phase='energy')
momentum_phase = output.filter(phase='momentum')

# Convert to pandas DataFrame
df = output.to_dataframe()
```

---

## Core Classes

### TrinityOutput

The main reader class for TRINITY output files.

```python
class TrinityOutput:
    """Reader for TRINITY simulation output files (.jsonl)."""

    # Properties
    filepath: Path          # Path to the data file
    model_name: str         # Model identifier
    keys: List[str]         # All available parameter names
    phases: List[str]       # Unique simulation phases
    t_min: float           # Minimum time in output [Myr]
    t_max: float           # Maximum time in output [Myr]
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `open(filepath)` | Class method to open a file |
| `get(key, as_array=True)` | Get parameter across all snapshots |
| `get_at_time(t, mode='interpolate')` | Get snapshot at specific time |
| `filter(phase, t_min, t_max)` | Filter snapshots by criteria |
| `info(verbose=False)` | Print file information |
| `to_dataframe()` | Convert to pandas DataFrame |

### Snapshot

Represents a single simulation timestep.

```python
@dataclass
class Snapshot:
    """A single simulation snapshot."""
    data: Dict[str, Any]      # All data for this timestep
    index: int                # Index in the output file
    is_interpolated: bool     # True if created via interpolation
    interpolation_time: float # Requested time (if interpolated)

    # Properties
    t_now: float              # Current time [Myr]
    phase: str                # Current simulation phase
```

**Usage:**

```python
# Access by index
snap = output[100]

# Access data
radius = snap['R2']
velocity = snap.get('v2', default=0.0)

# Check properties
print(snap.t_now, snap.phase)

# List available keys
print(snap.keys())
```

---

## Reading Data

### Opening Files

```python
from src._output.trinity_reader import TrinityOutput, load_output

# Method 1: Class method
output = TrinityOutput.open('/path/to/dictionary.jsonl')

# Method 2: Convenience function
output = load_output('/path/to/dictionary.jsonl')

# Supports both .jsonl (line-delimited) and .json formats
# Auto-detects format based on content
```

### Accessing Time Series

```python
# Get any parameter as numpy array
times = output.get('t_now')           # [Myr]
radii = output.get('R2')              # [pc]
energy = output.get('Eb')             # [erg]
pressure = output.get('Pb')           # [dyn/cm^2]

# For non-numeric data, disable array conversion
phases = output.get('current_phase', as_array=False)  # List of strings

# Access all available keys
print(output.keys)
```

### Getting Snapshots

```python
# By index
first_snap = output[0]
last_snap = output[-1]
range_of_snaps = output[10:20]

# Iterate over all snapshots
for snap in output:
    print(f"t={snap.t_now:.3f} Myr, R2={snap['R2']:.2f} pc")

# At specific time (interpolated by default)
snap = output.get_at_time(1.0)

# At specific time (closest actual snapshot)
snap = output.get_at_time(1.0, mode='closest')

# Get just one parameter at a time
radius_at_1myr = output.get_at_time(1.0, key='R2')
```

### Filtering Data

```python
# Filter by simulation phase
energy_phase = output.filter(phase='energy')
implicit_phase = output.filter(phase='implicit')

# Filter by time range
early_data = output.filter(t_max=1.0)        # t < 1 Myr
late_data = output.filter(t_min=3.0)         # t > 3 Myr
window = output.filter(t_min=1.0, t_max=2.0) # 1 < t < 2 Myr

# Combine filters
early_energy = output.filter(phase='energy', t_max=1.0)

# Filtered output is also a TrinityOutput
print(len(energy_phase))  # Number of snapshots in energy phase
```

---

## Interpolation

When requesting data at a time that doesn't exactly match a snapshot, the reader can interpolate:

```python
# Interpolated snapshot (default)
snap = output.get_at_time(0.5)
# Prints: "[TrinityOutput] Time t=5.00e-01 Myr not found in snapshots.
#         Interpolating from 5 neighbors..."

# Closest actual snapshot (no interpolation)
snap = output.get_at_time(0.5, mode='closest')
# Prints: "[TrinityOutput] Time t=5.00e-01 Myr not found in snapshots.
#         Returning closest snapshot at t=4.98e-01 Myr."

# Suppress messages
snap = output.get_at_time(0.5, quiet=True)

# Control interpolation neighbors
snap = output.get_at_time(0.5, n_neighbors=10)  # Use more neighbors
```

**Interpolation behavior by data type:**

| Type | Behavior |
|------|----------|
| Numeric scalars | Linear interpolation |
| Numeric arrays | Element-wise linear interpolation |
| Strings (e.g., phase) | Use closest snapshot |
| Booleans | Use closest snapshot |
| None values | Preserved as None |

---

## Path Utilities

### Finding Single Files

```python
from src._output.trinity_reader import find_data_file, find_data_path, resolve_data_input

# Find data file for a run name
base_dir = '/path/to/outputs'
path = find_data_file(base_dir, '1e7_sfe020_n1e4')
# Searches: {run}_modified/ first, then {run}/
# Prefers: dictionary.jsonl > dictionary.json

# Find data file with path resolution
path = find_data_path('/path/to/dictionary')
# Tries: .jsonl, .json, directory with dictionary inside

# Flexible input resolution
path = resolve_data_input('1e7_sfe020_n1e4', output_dir='/path/to/outputs')
# Accepts: folder name, folder path, or file path
```

### Batch Processing

```python
from src._output.trinity_reader import find_all_simulations

# Find all simulations in a directory (recursive)
sim_files = find_all_simulations('/path/to/outputs')
# Returns: List[Path] to dictionary.jsonl files

for data_path in sim_files:
    output = load_output(data_path)
    # Process each simulation...
```

### Grid Organization

For parameter sweep plots, organize simulations by cloud mass and SFE:

```python
from src._output.trinity_reader import (
    find_all_simulations,
    organize_simulations_for_grid,
    get_unique_ndens
)

# Find all simulations
sim_files = find_all_simulations('/path/to/outputs')

# Get unique density values
densities = get_unique_ndens(sim_files)
# Returns: ['1e3', '1e4'] (sorted)

# Organize into grid structure
organized = organize_simulations_for_grid(sim_files, ndens_filter='1e4')

# Access organized data
mCloud_list = organized['mCloud_list']  # ['1e5', '1e6', '1e7', ...]
sfe_list = organized['sfe_list']        # ['001', '010', '020', ...]
grid = organized['grid']                 # {(mCloud, sfe): Path}

# Plot grid
for mCloud in mCloud_list:
    for sfe in sfe_list:
        data_path = grid.get((mCloud, sfe))
        if data_path:
            output = load_output(data_path)
            # Plot this simulation...
```

---

## Available Parameters

Use `output.info(verbose=True)` for complete documentation. Key parameters:

### Dynamical Variables

| Parameter | Description | Units |
|-----------|-------------|-------|
| `t_now` | Current simulation time | Myr |
| `R2` | Outer bubble/shell radius | pc |
| `v2` | Shell expansion velocity | pc/Myr |
| `Eb` | Bubble thermal energy | erg |
| `T0` | Characteristic bubble temperature | K |
| `R1` | Inner bubble radius (wind termination shock) | pc |
| `Pb` | Bubble pressure | dyn/cm² |

### Forces

| Parameter | Description |
|-----------|-------------|
| `F_grav` | Gravitational force |
| `F_ram` | Ram pressure force (total) |
| `F_ram_wind` | Ram pressure from winds |
| `F_ram_SN` | Ram pressure from supernovae |
| `F_ion_out` | Ionization force (outward) |
| `F_rad` | Radiation pressure force |

### Cooling Parameters

| Parameter | Description |
|-----------|-------------|
| `cool_beta` | Pressure evolution: β = -(t/Pb)(dPb/dt) |
| `cool_delta` | Temperature evolution parameter δ |

### Feedback Luminosities

| Parameter | Description | Units |
|-----------|-------------|-------|
| `Lmech_W` | Mechanical luminosity from winds | erg/Myr |
| `Lmech_SN` | Mechanical luminosity from supernovae | erg/Myr |
| `Qi` | Ionizing photon rate | photons/s |

### Shell Properties

| Parameter | Description | Units |
|-----------|-------------|-------|
| `shell_mass` | Shell mass | Msun |
| `shell_thickness` | Shell thickness | pc |
| `shell_n0` | Shell number density at inner edge | cm⁻³ |

---

## Examples

### Example 1: Basic Plotting

```python
import matplotlib.pyplot as plt
from src._output.trinity_reader import load_output

output = load_output('/path/to/dictionary.jsonl')

t = output.get('t_now')
R2 = output.get('R2')

plt.figure()
plt.plot(t, R2)
plt.xlabel('Time [Myr]')
plt.ylabel('Shell Radius [pc]')
plt.title(output.model_name)
plt.show()
```

### Example 2: Phase-Colored Plot

```python
import matplotlib.pyplot as plt
import numpy as np
from src._output.trinity_reader import load_output

output = load_output('/path/to/dictionary.jsonl')

t = output.get('t_now')
R2 = output.get('R2')
phases = output.get('current_phase', as_array=False)

phase_colors = {'energy': 'blue', 'momentum': 'red', 'transition': 'orange'}
colors = [phase_colors.get(p, 'gray') for p in phases]

plt.figure()
plt.scatter(t, R2, c=colors, s=1)
plt.xlabel('Time [Myr]')
plt.ylabel('Shell Radius [pc]')
plt.show()
```

### Example 3: Force Balance Analysis

```python
from src._output.trinity_reader import load_output
import numpy as np

output = load_output('/path/to/dictionary.jsonl')

# Get all forces
F_grav = np.abs(output.get('F_grav'))
F_ram = np.abs(output.get('F_ram'))
F_ion = np.abs(output.get('F_ion_out'))
F_rad = np.abs(output.get('F_rad'))

# Find dominant force at each timestep
forces = np.vstack([F_grav, F_ram, F_ion, F_rad])
dominant = np.argmax(forces, axis=0)

labels = ['Gravity', 'Ram', 'Ionization', 'Radiation']
for i, label in enumerate(labels):
    frac = np.sum(dominant == i) / len(dominant) * 100
    print(f"{label}: {frac:.1f}% of timesteps")
```

### Example 4: Batch Processing Grid

```python
from src._output.trinity_reader import (
    find_all_simulations, organize_simulations_for_grid, load_output
)

# Find and organize simulations
sim_files = find_all_simulations('/path/to/outputs')
organized = organize_simulations_for_grid(sim_files, ndens_filter='1e4')

# Calculate final radius for each simulation
results = {}
for mCloud in organized['mCloud_list']:
    for sfe in organized['sfe_list']:
        path = organized['grid'].get((mCloud, sfe))
        if path:
            output = load_output(path)
            final_R2 = output[-1]['R2']
            results[(mCloud, sfe)] = final_R2

print(results)
```

---

## API Reference

### Functions

| Function | Description |
|----------|-------------|
| `load_output(filepath)` | Load a TRINITY output file |
| `read(filepath)` | Alias for `load_output` |
| `find_data_file(base_dir, run_name)` | Find data file for a run |
| `find_data_path(base_path)` | Find data file with extension resolution |
| `resolve_data_input(data_input, output_dir)` | Flexible path resolution |
| `find_all_simulations(base_dir)` | Find all simulations recursively |
| `parse_simulation_params(folder_name)` | Extract mCloud/sfe/ndens from name |
| `get_unique_ndens(sim_files)` | Get unique density values |
| `organize_simulations_for_grid(sim_files, ndens_filter)` | Organize for grid plots |

### TrinityOutput Methods

| Method | Description |
|--------|-------------|
| `open(filepath)` | Class method to open a file |
| `get(key, as_array=True)` | Get parameter across all snapshots |
| `get_at_time(t, key, mode, n_neighbors, quiet)` | Get snapshot at time |
| `filter(phase, t_min, t_max)` | Filter snapshots |
| `info(verbose=False)` | Print file information |
| `to_dataframe()` | Convert to pandas DataFrame |

### Snapshot Methods

| Method | Description |
|--------|-------------|
| `__getitem__(key)` | Access data by key: `snap['R2']` |
| `get(key, default)` | Get with default value |
| `keys()` | List all available keys |

---

## Notes

### Snapshot Consistency

As of January 2026, TRINITY snapshots are saved BEFORE ODE integration, ensuring all values in a snapshot correspond to the same timestamp (`t_now`). This includes:
- Dynamical variables (R2, v2, Eb, T0)
- Feedback properties
- Shell and bubble properties
- Forces
- Beta-delta residuals

### File Formats

- **`.jsonl`** (preferred): Line-delimited JSON, one snapshot per line
- **`.json`**: Legacy format with all snapshots in one object

The reader automatically detects and handles both formats.

### Memory Efficiency

All snapshots are loaded into memory on open. For very large files, consider:
- Filtering early: `output.filter(t_max=5.0)` before heavy processing
- Using `to_dataframe()` for pandas-based analysis with lazy evaluation
