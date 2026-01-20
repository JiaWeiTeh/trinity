# TRINITY Logging System - Complete Guide

## Quick Start

### 1. Add to your parameter file (.param):

```
# --- Logging parameters
log_level    INFO          # DEBUG, INFO, WARNING, ERROR, CRITICAL
log_console  True          # Print to terminal
log_file     True          # Write to .log file
log_colors   True          # Use colored output
```

### 2. In main.py (set up once at start):

```python
from src._functions.logging_setup import setup_logging_from_params

def start_expansion(params):
    # Set up logging (reads from params)
    logger = setup_logging_from_params(params)

    logger.info("TRINITY simulation starting")
    # ... rest of code ...
```

### 3. In any other module:

```python
import logging

# At the top of file, after other imports
logger = logging.getLogger(__name__)

def my_function():
    logger.debug("Detailed debugging info")
    logger.info("Important event")
    logger.warning("Something unexpected")
    logger.error("Error occurred")
```

---

## Understanding Log Levels

### What is `log_level`?

**Log level** controls **how much detail** you see in the logs. It's like a **filter** that decides which messages get shown.

Think of it like a **volume knob** for logging:
- `DEBUG` = MAXIMUM volume (see everything)
- `INFO` = Medium-high volume (see important events)
- `WARNING` = Medium volume (only warnings and errors)
- `ERROR` = Low volume (only errors)
- `CRITICAL` = MINIMUM volume (only critical crashes)

### The 5 Log Levels (from most to least verbose):

#### 1. **DEBUG** (10)
**When to use**: During development, debugging, or when you need to see **everything**

**What you'll see**:
- Variable values at each timestep
- Loop iterations
- Intermediate calculations
- Function entry/exit
- All INFO, WARNING, ERROR, CRITICAL messages

**Example output**:
```
2026-01-08 15:30:01 | DEBUG    | src.phase1_energy.run_energy_phase | Timestep 150: R2=12.5 pc, v2=180 pc/Myr
2026-01-08 15:30:01 | DEBUG    | src.shell_structure.shell_structure | Integrating shell from r=10.2 to r=15.8 pc
2026-01-08 15:30:01 | DEBUG    | src.cooling.net_coolingcurve | Interpolating cooling at T=1.5e6 K
2026-01-08 15:30:02 | INFO     | src.phase1_energy.run_energy_phase | Energy phase complete
```

**Warning**: DEBUG produces A LOT of output! Use only when debugging specific issues.

#### 2. **INFO** (20) ⭐ RECOMMENDED DEFAULT
**When to use**: Normal simulation runs

**What you'll see**:
- Phase transitions (energy → implicit → transition → momentum)
- Major events (bubble bursts, cloud edge reached)
- Initialization (SB99 data loaded, cooling curves loaded)
- Completion messages
- All WARNING, ERROR, CRITICAL messages

**Example output**:
```
2026-01-08 15:30:00 | INFO     | src.main | TRINITY simulation starting
2026-01-08 15:30:00 | INFO     | src.main | Output directory: outputs/my_simulation
2026-01-08 15:30:01 | INFO     | src.main | SB99 data loaded
2026-01-08 15:30:01 | INFO     | src.phase1_energy.run_energy_phase | Entering energy-driven phase
2026-01-08 15:30:45 | INFO     | src.phase1_energy.run_energy_phase | Energy phase complete after 150 iterations
2026-01-08 15:31:00 | WARNING  | src.cooling.net_coolingcurve | Temperature below minimum, clamping to 1e4 K
2026-01-08 15:32:00 | INFO     | src.main | Simulation complete
```

**Recommendation**: Use INFO for production runs. Clean, informative, not overwhelming.

#### 3. **WARNING** (30)
**When to use**: When you only care about **potential problems**, not normal operation

**What you'll see**:
- Values clamped to limits (temperature, density)
- Fallback to default behavior
- Unusual but non-critical conditions
- All ERROR, CRITICAL messages

**Example output**:
```
2026-01-08 15:30:01 | WARNING  | src.cooling.net_coolingcurve | Temperature T=5.2e3 K below minimum 1e4 K, clamping
2026-01-08 15:30:15 | WARNING  | src.sb99.update_feedback | Wind momentum rate near zero at t=15.2 Myr, setting vWind=0
2026-01-08 15:30:20 | ERROR    | src.shell_structure.shell_structure | Mass conservation violated by 15%
```

**Use case**: Production runs where you only want to see warnings/errors, not normal events.

#### 4. **ERROR** (40)
**When to use**: When you only want to see **actual errors** (but simulation continues)

**What you'll see**:
- Calculation failures
- Unexpected conditions
- Recoverable errors
- All CRITICAL messages

**Example output**:
```
2026-01-08 15:30:20 | ERROR    | src.shell_structure.shell_structure | Mass conservation violated by 15%
2026-01-08 15:30:25 | ERROR    | src.bubble_structure.bubble_luminosity | Temperature solver did not converge, retrying
```

**Use case**: Final production runs where you only want to know if something went wrong.

#### 5. **CRITICAL** (50)
**When to use**: Only when you want to see **simulation-stopping errors**

**What you'll see**:
- Unrecoverable errors
- Fatal failures
- Simulation crashes

**Example output**:
```
2026-01-08 15:30:30 | CRITICAL | src.main | Simulation failed: NaN values detected in all variables
```

**Use case**: Rarely used; ERROR level usually sufficient.

---

## Log Level Hierarchy

**Key concept**: Setting a log level shows **that level AND everything more severe**.

```
DEBUG ────> Shows: DEBUG, INFO, WARNING, ERROR, CRITICAL (ALL messages)
   ↓
INFO ─────> Shows: INFO, WARNING, ERROR, CRITICAL
   ↓
WARNING ──> Shows: WARNING, ERROR, CRITICAL
   ↓
ERROR ────> Shows: ERROR, CRITICAL
   ↓
CRITICAL ─> Shows: CRITICAL only
```

### Examples:

```python
# Set log level to INFO
log_level = INFO

# These will show:
logger.info("Phase started")         ✅ Shows
logger.warning("Temperature clamped") ✅ Shows
logger.error("Calculation failed")    ✅ Shows
logger.critical("Simulation crashed") ✅ Shows

# This will NOT show:
logger.debug("R2 = 12.5 pc")          ❌ Hidden (below INFO)
```

---

## Other Logging Parameters

### `log_console` (True/False)

**Controls**: Whether log messages print to **terminal** (stdout)

```
log_console = True   → Messages print to screen (see them during run)
log_console = False  → No terminal output (silent run)
```

**Use case**:
- `True`: Interactive runs where you want to see progress
- `False`: Batch runs where you only want file output

### `log_file` (True/False)

**Controls**: Whether log messages save to **.log file** in output directory

```
log_file = True   → Creates trinity_YYYYMMDD_HHMMSS.log in path2output
log_file = False  → No log file created
```

**Use case**:
- `True`: Always recommended! Permanent record of simulation
- `False`: Testing/debugging only

### `log_colors` (True/False)

**Controls**: Color-coded terminal output

```
log_colors = True   → DEBUG=cyan, INFO=green, WARNING=yellow, ERROR=red, CRITICAL=magenta
log_colors = False  → Plain text (all white)
```

**Colors**:
- 🔵 **DEBUG**: Cyan (easy to distinguish from normal output)
- 🟢 **INFO**: Green (positive, normal operation)
- 🟡 **WARNING**: Yellow (caution, but not critical)
- 🔴 **ERROR**: Red (problem occurred)
- 🟣 **CRITICAL**: Magenta (severe, simulation-stopping)

**Use case**:
- `True`: Interactive terminal (easier to spot errors)
- `False`: Plain terminals, file redirection, or color-blind accessibility

---

## Common Configurations

### 1. **Development/Debugging** (maximum detail, console only)
```
log_level    DEBUG
log_console  True
log_file     False
log_colors   True
```

### 2. **Normal Run** (standard detail, console + file) ⭐ RECOMMENDED
```
log_level    INFO
log_console  True
log_file     True
log_colors   True
```

### 3. **Production Run** (warnings only, file output)
```
log_level    WARNING
log_console  False
log_file     True
log_colors   False
```

### 4. **Silent Run** (errors only, no console spam)
```
log_level    ERROR
log_console  False
log_file     True
log_colors   False
```

### 5. **Quick Test** (maximum detail, no file)
```
log_level    DEBUG
log_console  True
log_file     False
log_colors   True
```

---

## Where Do Logs Go?

### Console Output:
- Prints to terminal (stdout) while simulation runs
- Disappears when terminal closes
- Can redirect: `python run.py > output.txt 2>&1`

### File Output:
- Saved to: `{path2output}/trinity_YYYYMMDD_HHMMSS.log`
- Example: `outputs/my_simulation/trinity_20260108_153045.log`
- Permanent record
- Plain text (no colors in file, for readability)

---

## How to Use in Your Code

### In main.py (set up ONCE at start):

```python
from src._functions.logging_setup import setup_logging_from_params
import logging

def start_expansion(params):
    # Initialize logging system
    logger = setup_logging_from_params(params)

    logger.info("=== TRINITY Simulation Starting ===")
    logger.info(f"Model: {params['model_name'].value}")
    logger.info(f"Output: {params['path2output'].value}")

    # ... load SB99 data ...
    logger.info("SB99 data loaded successfully")

    # ... load cooling curves ...
    logger.info("Cooling curves loaded successfully")

    # ... run phases ...
    logger.info("All phases complete")
    logger.info("=== Simulation Finished ===")
```

### In ANY other module (phase files, utilities, etc.):

```python
import logging

# Get module-specific logger (ONCE at top of file)
logger = logging.getLogger(__name__)

def run_energy_phase(params):
    logger.info("Entering energy-driven phase")

    for ii, t in enumerate(time_array):
        logger.debug(f"Timestep {ii}: t={t:.6f} Myr, R2={R2:.3f} pc, v2={v2:.2f} pc/Myr")

        if T < T_min:
            logger.warning(f"Temperature T={T:.2e} K below minimum {T_min:.2e} K, clamping")
            T = T_min

        try:
            # ... do calculation ...
            pass
        except Exception as e:
            logger.error(f"Calculation failed at timestep {ii}: {e}")
            raise

    logger.info(f"Energy phase complete after {ii+1} timesteps")
```

### Logging Best Practices:

1. **Use appropriate levels**:
   - `logger.debug()`: Variable values, loop iterations, temporary calculations
   - `logger.info()`: Major events, phase transitions, completion
   - `logger.warning()`: Unexpected but recoverable (clamped values, fallbacks)
   - `logger.error()`: Calculation failures, invalid state
   - `logger.critical()`: Unrecoverable errors, simulation must stop

2. **Be specific**:
   ```python
   # ❌ Bad: Vague
   logger.warning("Problem detected")

   # ✅ Good: Specific
   logger.warning(f"Temperature T={T:.2e} K below minimum {T_min:.2e} K, clamping to minimum")
   ```

3. **Include values**:
   ```python
   # ❌ Bad: No context
   logger.error("Convergence failed")

   # ✅ Good: Includes values
   logger.error(f"Convergence failed after {iterations} iterations, residual={residual:.3e} > tolerance={tol:.3e}")
   ```

4. **One logger per module**:
   ```python
   # At top of file, OUTSIDE functions:
   logger = logging.getLogger(__name__)

   # Then use throughout file:
   def function1():
       logger.info("Function 1")

   def function2():
       logger.debug("Function 2")
   ```

---

## Example Log Output

### With `log_level = INFO`, `log_console = True`, `log_colors = True`:

```
2026-01-08 15:30:00 | INFO     | src.main | Log file: outputs/test_sim/trinity_20260108_153000.log
2026-01-08 15:30:00 | INFO     | src.main | === TRINITY Simulation Starting ===
2026-01-08 15:30:00 | INFO     | src.main | Model: test_simulation
2026-01-08 15:30:00 | INFO     | src.main | Output: outputs/test_sim
2026-01-08 15:30:01 | INFO     | src.sb99.read_SB99 | Reading SB99 file: 1e6cluster_rot_Z0014_BH120.txt
2026-01-08 15:30:01 | INFO     | src.sb99.read_SB99 | SB99 data loaded: 201 time points from 0.00 to 40.00 Myr
2026-01-08 15:30:02 | INFO     | src.cooling.read_cloudy | Loading CLOUDY cooling tables
2026-01-08 15:30:03 | INFO     | src.main | Initialization complete
2026-01-08 15:30:03 | INFO     | src.phase1_energy.run_energy_phase | Entering energy-driven phase
2026-01-08 15:30:03 | INFO     | src.phase1_energy.run_energy_phase | Initial: R2=1.50 pc, v2=200.5 pc/Myr, Eb=5.2e50 erg
2026-01-08 15:30:15 | WARNING  | src.cooling.net_coolingcurve | Temperature T=8.5e3 K below minimum 1.0e4 K, clamping
2026-01-08 15:30:45 | INFO     | src.phase1_energy.run_energy_phase | Energy phase complete: 150 timesteps, 0.0028 Myr
2026-01-08 15:30:45 | INFO     | src.phase1b_energy_implicit.run_energy_implicit_phase | Entering implicit energy phase
2026-01-08 15:31:30 | INFO     | src.phase1b_energy_implicit.run_energy_implicit_phase | Implicit phase complete
2026-01-08 15:31:30 | INFO     | src.phase1c_transition.run_transition_phase | Entering transition phase
2026-01-08 15:32:00 | INFO     | src.phase1c_transition.run_transition_phase | Transition phase complete
2026-01-08 15:32:00 | INFO     | src.phase2_momentum.run_momentum_phase | Entering momentum-driven phase
2026-01-08 15:35:00 | INFO     | src.phase2_momentum.run_momentum_phase | Momentum phase complete
2026-01-08 15:35:00 | INFO     | src.main | === Simulation Finished ===
2026-01-08 15:35:00 | INFO     | src.main | Total runtime: 5.0 minutes
```

---

## Summary

| Parameter | What It Does | Recommended Value |
|-----------|--------------|-------------------|
| **log_level** | How much detail to show | `INFO` (normal), `DEBUG` (debugging) |
| **log_console** | Print to terminal? | `True` (interactive), `False` (batch) |
| **log_file** | Save to .log file? | `True` (always!) |
| **log_colors** | Colored output? | `True` (terminal), `False` (plain text) |

**Golden Rule**: Start with `log_level = INFO`, adjust as needed. Use `DEBUG` when things go wrong!
