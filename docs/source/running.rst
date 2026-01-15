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

TRINITY provides flexible logging to help monitor simulation progress.

Logging Parameters
^^^^^^^^^^^^^^^^^^

Configure logging in your parameter file:

.. code-block:: text

    log_level     INFO      # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_console   True      # Print to terminal
    log_file      True      # Write to .log file
    log_colors    True      # Color-coded terminal output

Log Levels
^^^^^^^^^^

================  ===============================================================
Level             Description
================  ===============================================================
``DEBUG``         Most detailed - includes internal calculations
``INFO``          General progress information (recommended for most users)
``WARNING``       Warnings about potential issues
``ERROR``         Error conditions that may affect results
``CRITICAL``      Severe errors that halt execution
================  ===============================================================

Color Coding (Terminal)
^^^^^^^^^^^^^^^^^^^^^^^

When ``log_colors = True``, messages are color-coded:

- **Cyan**: DEBUG
- **Green**: INFO
- **Yellow**: WARNING
- **Red**: ERROR
- **Magenta**: CRITICAL


Verbosity Control
-----------------

The ``verbose`` parameter controls the amount of terminal output:

.. code-block:: text

    verbose    1    # Minimal output
    verbose    2    # Standard output (default)
    verbose    3    # Detailed output


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
