.. highlight:: rest

.. _sec-running:

Running TRINITY
===============

This section covers how to run TRINITY simulations, from basic single runs to parallel parameter sweeps.

Quick Start
-----------

The simplest way to run TRINITY is with a minimal parameter file. Create a file ``my_run.param``:

.. code-block:: text

    model_name    my_first_run
    mCloud        1e6
    sfe           0.01

Then execute from the TRINITY root directory:

.. code-block:: console

    python run.py param/my_run.param

That's it! TRINITY will use default values for all unspecified parameters.


Single Simulation Runs
----------------------

Command Syntax
^^^^^^^^^^^^^^

.. code-block:: console

    python run.py <path_to_parameter_file>

The parameter file path can be absolute or relative to the TRINITY root directory.

**Examples:**

.. code-block:: console

    # Using a file in the param/ directory
    python run.py param/example.param

    # Using an absolute path
    python run.py /home/user/my_params/custom.param

Output Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^

TRINITY creates the following structure in your output directory (set by ``path2output``):

.. code-block:: text

    path2output/
    ├── {model_name}_summary.txt    # Human-readable parameter summary
    ├── {model_name}.log            # Log file (if log_file = True)
    └── {model_name}.json           # Simulation output data

If ``path2output`` is set to ``def_dir`` (default), outputs are written to the directory where TRINITY is executed.


Parameter Sweep Runs
--------------------

TRINITY supports running multiple simulations with different parameter combinations using the sweep system.

Sweep Syntax
^^^^^^^^^^^^

In a sweep parameter file, use list notation ``[val1, val2, ...]`` for parameters you want to vary:

.. code-block:: text

    # Sweep parameters - will generate all combinations
    mCloud    [1e5, 1e7, 1e8]
    sfe       [0.01, 0.10]
    nCore     [1e2, 1e3]

    # Fixed parameters - constant across all runs
    dens_profile    densPL
    densPL_alpha    0
    path2output     outputs/my_sweep

This generates a Cartesian product: 3 x 2 x 2 = 12 total simulations.

Running Sweeps
^^^^^^^^^^^^^^

Use ``run_sweep.py`` to execute parameter sweeps:

.. code-block:: console

    # Preview combinations without running (dry run)
    python run_sweep.py param/sweep.param --dry-run

    # Run with 4 parallel workers
    python run_sweep.py param/sweep.param --workers 4

    # Run with automatic worker detection (default: CPU count - 1, max 8)
    python run_sweep.py param/sweep.param

**Command-line options:**

==================  ============================================================
Option              Description
==================  ============================================================
``--dry-run``       Preview all parameter combinations without executing
``--workers N``     Number of parallel processes (default: auto-detect)
==================  ============================================================

Auto-Generated Run Names
^^^^^^^^^^^^^^^^^^^^^^^^

Each combination automatically receives a descriptive name following this convention:

.. code-block:: text

    {mCloud}_sfe{sfe*100:03d}_n{nCore}

**Examples:**

- ``1e5_sfe001_n1e2`` for mCloud=1e5, sfe=0.01, nCore=1e2
- ``1e7_sfe010_n1e3`` for mCloud=1e7, sfe=0.10, nCore=1e3

Output files are organized into subdirectories:

.. code-block:: text

    outputs/my_sweep/
    ├── 1e5_sfe001_n1e2/
    │   ├── 1e5_sfe001_n1e2_summary.txt
    │   └── 1e5_sfe001_n1e2.json
    ├── 1e5_sfe001_n1e3/
    │   └── ...
    └── sweep_report.json           # Summary of all runs


Logging Configuration
---------------------

TRINITY provides a flexible logging system to help monitor simulation progress and debug issues.

Logging Parameters
^^^^^^^^^^^^^^^^^^

Configure logging in your parameter file:

.. code-block:: text

    log_level     INFO      # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_console   True      # Print to terminal
    log_file      True      # Write to .log file
    log_colors    True      # Color-coded terminal output

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``log_level``
     - ``INFO``
     - Controls how much detail you see. Think of it as a volume knob for logging.
   * - ``log_console``
     - ``True``
     - Print log messages to terminal during simulation.
   * - ``log_file``
     - ``True``
     - Save log messages to ``{path2output}/trinity_YYYYMMDD_HHMMSS.log``.
   * - ``log_colors``
     - ``True``
     - Color-code terminal output by severity level.


Log Levels
^^^^^^^^^^

Setting a log level shows **that level and everything more severe**:

.. raw:: html

   <style>
   .log-debug { color: #00CED1; font-weight: bold; }
   .log-info { color: #32CD32; font-weight: bold; }
   .log-warning { color: #FFA500; font-weight: bold; }
   .log-error { color: #FF4444; font-weight: bold; }
   .log-critical { color: #FF00FF; font-weight: bold; }
   </style>

.. raw:: html

   <table style="width:100%; border-collapse: collapse; margin: 1em 0;">
   <tr style="background: #f0f0f0;">
     <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Level</th>
     <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Shows</th>
     <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Use Case</th>
   </tr>
   <tr>
     <td style="padding: 8px; border: 1px solid #ddd;"><span class="log-debug">DEBUG</span></td>
     <td style="padding: 8px; border: 1px solid #ddd;">All messages</td>
     <td style="padding: 8px; border: 1px solid #ddd;">Development, debugging specific issues</td>
   </tr>
   <tr>
     <td style="padding: 8px; border: 1px solid #ddd;"><span class="log-info">INFO</span> ⭐</td>
     <td style="padding: 8px; border: 1px solid #ddd;">INFO + WARNING + ERROR + CRITICAL</td>
     <td style="padding: 8px; border: 1px solid #ddd;"><strong>Recommended</strong> - Normal simulation runs</td>
   </tr>
   <tr>
     <td style="padding: 8px; border: 1px solid #ddd;"><span class="log-warning">WARNING</span></td>
     <td style="padding: 8px; border: 1px solid #ddd;">WARNING + ERROR + CRITICAL</td>
     <td style="padding: 8px; border: 1px solid #ddd;">Production runs, only see potential problems</td>
   </tr>
   <tr>
     <td style="padding: 8px; border: 1px solid #ddd;"><span class="log-error">ERROR</span></td>
     <td style="padding: 8px; border: 1px solid #ddd;">ERROR + CRITICAL</td>
     <td style="padding: 8px; border: 1px solid #ddd;">Silent runs, only see actual errors</td>
   </tr>
   <tr>
     <td style="padding: 8px; border: 1px solid #ddd;"><span class="log-critical">CRITICAL</span></td>
     <td style="padding: 8px; border: 1px solid #ddd;">CRITICAL only</td>
     <td style="padding: 8px; border: 1px solid #ddd;">Only simulation-stopping errors</td>
   </tr>
   </table>


What Each Level Shows
"""""""""""""""""""""

.. raw:: html

   <p><span class="log-debug">DEBUG</span> - Variable values, loop iterations, intermediate calculations, function entry/exit</p>
   <p><span class="log-info">INFO</span> - Phase transitions, major events (bubble bursts, cloud edge reached), initialization, completion</p>
   <p><span class="log-warning">WARNING</span> - Values clamped to limits, fallback behavior, unusual but non-critical conditions</p>
   <p><span class="log-error">ERROR</span> - Calculation failures, unexpected conditions, recoverable errors</p>
   <p><span class="log-critical">CRITICAL</span> - Unrecoverable errors, fatal failures, simulation crashes</p>


Common Configurations
^^^^^^^^^^^^^^^^^^^^^

**Development/Debugging** (maximum detail):

.. code-block:: text

    log_level    DEBUG
    log_console  True
    log_file     False
    log_colors   True

**Normal Run** (recommended):

.. code-block:: text

    log_level    INFO
    log_console  True
    log_file     True
    log_colors   True

**Production/Batch Run** (minimal output):

.. code-block:: text

    log_level    WARNING
    log_console  False
    log_file     True
    log_colors   False


Example Log Output
^^^^^^^^^^^^^^^^^^

With ``log_level = INFO``:

.. code-block:: text

    2026-01-08 15:30:00 | INFO     | src.main | === TRINITY Simulation Starting ===
    2026-01-08 15:30:00 | INFO     | src.main | Model: test_simulation
    2026-01-08 15:30:01 | INFO     | src.sb99.read_SB99 | SB99 data loaded: 201 time points
    2026-01-08 15:30:03 | INFO     | src.phase1_energy | Entering energy-driven phase
    2026-01-08 15:30:15 | WARNING  | src.cooling | Temperature below minimum, clamping to 1e4 K
    2026-01-08 15:30:45 | INFO     | src.phase1_energy | Energy phase complete: 150 timesteps
    2026-01-08 15:35:00 | INFO     | src.main | === Simulation Finished ===


Output Formats
--------------

JSON Output
^^^^^^^^^^^

The primary output format is JSON (``output_format = JSON``), containing:

- All input parameters with metadata
- Time-evolution arrays (radius, velocity, temperature, etc.)
- Derived quantities (forces, luminosities, masses)

Snapshot System
^^^^^^^^^^^^^^^

TRINITY uses an append-only JSONL (JSON Lines) format for snapshots, where each line represents one timestep. This provides:

- O(1) write performance for large simulations
- Easy parsing and streaming of results
- Automatic array simplification for efficiency


Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**"Parameter not found in default.param"**
    Your parameter file contains a typo or uses an outdated parameter name.
    Check :ref:`sec-parameters` for valid parameter names.

**"Only solar metallicity supported"**
    Currently, TRINITY only supports ``ZCloud = 1`` (solar metallicity).

**"Invalid density profile"**
    The ``dens_profile`` parameter must be either ``densPL`` or ``densBE``.

**Output directory not created**
    Ensure the parent directory exists. TRINITY creates the final directory but not parent paths.

Getting Help
^^^^^^^^^^^^

For issues and feature requests, visit:
https://github.com/JiaWeiTeh/trinity/issues
