# TRINITY Output Module

This module provides tools for reading and analyzing TRINITY simulation outputs.

## TrinityOutput Reader

The `TrinityOutput` class provides a clean, Pythonic API for accessing TRINITY simulation data, similar to `astropy.io.fits`.

### Quick Start

```python
from src._output.trinity_reader import TrinityOutput

# Open an output file
output = TrinityOutput.open('path/to/simulation.jsonl')

# Get summary information
output.info()              # Quick summary
output.info(verbose=True)  # Detailed parameter documentation

# Extract time series as numpy arrays
t = output.get('t_now')      # Time [Myr]
R2 = output.get('R2')        # Outer radius [pc]
v2 = output.get('v2')        # Velocity [pc/Myr]
Eb = output.get('Eb')        # Bubble energy [erg]

# For non-numeric data
phase = output.get('current_phase', as_array=False)

# Access individual snapshots
snap = output[100]           # Get snapshot 100
snap['R2']                   # Get R2 from that snapshot

# Get snapshot closest to a time
snap = output.get_at_time(1.0)  # Snapshot closest to 1 Myr

# Filter by phase or time
implicit = output.filter(phase='implicit')
early = output.filter(t_max=0.1)
combined = output.filter(phase='implicit', t_min=0.05, t_max=0.5)

# Convert to pandas DataFrame
df = output.to_dataframe()
```

### From Plotting Scripts

In `src/_plots/` scripts:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_file

data_path = find_data_file(BASE_DIR, '1e7_sfe020_n1e4')
output = load_output(data_path)

t = output.get('t_now')
R2 = output.get('R2')
```

## Key Output Parameters

### Dynamical Variables
- `t_now`: Current simulation time [Myr]
- `R2`: Outer bubble/shell radius [pc]
- `v2`: Shell expansion velocity [pc/Myr]
- `Eb`: Bubble thermal energy [erg]
- `T0`: Characteristic bubble temperature [K]
- `R1`: Inner bubble radius (wind shock) [pc]
- `Pb`: Bubble pressure [dyn/cm^2]

### Cooling Parameters
- `cool_beta`: Pressure evolution parameter
- `cool_delta`: Temperature evolution parameter

### Forces
- `F_grav`: Gravitational force
- `F_ram`: Ram pressure force
- `F_ion_out`: Ionization force (outward)
- `F_rad`: Radiation pressure force

### Residual Diagnostics (Beta-Delta Solver)
- `residual_Edot1_guess`: Edot from beta [erg/Myr]
- `residual_Edot2_guess`: Edot from energy balance [erg/Myr]
- `residual_T1_guess`: Bubble temperature [K]
- `residual_T2_guess`: Target temperature T0 [K]

Use `output.info(verbose=True)` to see all available parameters with documentation.

## Snapshot Consistency

As of January 2026, TRINITY snapshots are saved **BEFORE** ODE integration, ensuring all values in a snapshot correspond to the same timestamp (`t_now`). This includes:

- Dynamical variables: t_now, R2, v2, Eb, T0
- Feedback properties: Lmech, pdot, etc.
- Shell structure: shell_mass, shell_thickness, etc.
- Bubble structure: bubble_Tavg, bubble_mass, etc.
- Forces: F_grav, F_ram, F_ion, F_rad
- Cooling parameters: beta, delta
- Residual diagnostics

This consistency is critical for accurate analysis of correlations between variables.

## File Formats

TRINITY supports two output formats:

1. **JSONL** (recommended): One JSON object per line
   - Efficient streaming reads
   - Easy to append during simulation
   - File extension: `.jsonl`

2. **JSON** (legacy): Nested dictionary structure
   - Format: `{"0": {...}, "1": {...}, ...}`
   - File extension: `.json`

The TrinityOutput reader automatically detects and handles both formats.

## Examples

See `example_scripts/` for comprehensive examples:

- `example_reader_overview.py`: Full demonstration of reader features
- `example_plot_radius_vs_time.py`: Plotting examples

## See Also

- `src/_plots/paper_*.py`: Paper figure scripts using TrinityOutput
