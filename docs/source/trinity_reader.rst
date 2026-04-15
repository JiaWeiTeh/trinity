.. highlight:: rest

.. _sec-trinity-reader:

Output Reader API
=================

The ``trinity_reader`` module provides a Python interface for
loading and inspecting TRINITY simulation output. It sits on top of
the ``DescribedDict`` snapshot format described in :ref:`sec-running`
and hides the JSONL layout, the distinction between ``.jsonl`` and
legacy ``.json`` files, and the per-key unit metadata behind a small
set of high-level classes.

Two objects do most of the work. A :class:`TrinityOutput` represents
an entire simulation and exposes time-series access, filtering by
phase or time window, and conversion to a pandas ``DataFrame``. A
:class:`Snapshot` represents a single time step and behaves like a
dictionary keyed by parameter name. A companion utility,
``find_all_simulations``, walks a directory tree of sweep output
and returns the paths of every ``dictionary.jsonl`` it finds, which
is convenient for building grid plots across a parameter sweep.

Downstream plotting and analysis scripts under ``src/_plots/`` and
``src/_calc/`` consume their input exclusively through this module.

.. seealso::

   - :ref:`sec-running` — *Output Data Model* section — describes the on-disk
     JSONL layout, the ``DescribedDict`` / ``DescribedItem`` objects, the
     save/flush workflow, and the low-level ``DescribedDict.load_snapshot``
     API that the reader sits on top of.
   - :ref:`sec-parameters` — full list of parameter names and units that can
     appear as keys in a snapshot.
   - :ref:`sec-visualization` — ready-made plotting scripts that consume
     reader output.


Quick Start
-----------

.. code-block:: python

    from src._output.trinity_reader import load_output

    # Load a simulation
    output = load_output('/path/to/dictionary.jsonl')

    # Get basic info
    output.info()

    # Access time series as numpy arrays
    times = output.get('t_now')      # [Myr]
    radii = output.get('R2')         # [pc]
    velocity = output.get('v2')      # [pc/Myr]

    # Get snapshot at specific time
    snap = output.get_at_time(1.0)   # interpolated
    print(snap['R2'], snap['v2'])

    # Filter by phase
    energy_data = output.filter(phase='energy')

    # Convert to pandas DataFrame
    df = output.to_dataframe()


Core Classes
------------

TrinityOutput
^^^^^^^^^^^^^

The main reader class for TRINITY output files.

.. code-block:: python

    from src._output.trinity_reader import TrinityOutput

    output = TrinityOutput.open('dictionary.jsonl')

**Properties:**

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Property
     - Type
     - Description
   * - ``filepath``
     - Path
     - Path to the data file
   * - ``model_name``
     - str
     - Model identifier from first snapshot
   * - ``keys``
     - List[str]
     - All available parameter names
   * - ``phases``
     - List[str]
     - Unique simulation phases found
   * - ``t_min``
     - float
     - Minimum time in output [Myr]
   * - ``t_max``
     - float
     - Maximum time in output [Myr]

**Methods:**

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Method
     - Description
   * - ``open(filepath)``
     - Class method to open a file
   * - ``get(key, as_array=True)``
     - Get parameter across all snapshots as numpy array
   * - ``get_at_time(t, mode='interpolate')``
     - Get snapshot at specific time (interpolated or closest)
   * - ``filter(phase, t_min, t_max)``
     - Filter snapshots by phase or time range
   * - ``info(verbose=False)``
     - Print file information and parameters
   * - ``to_dataframe()``
     - Convert to pandas DataFrame

Snapshot
^^^^^^^^

Represents a single simulation timestep.

.. code-block:: python

    # Access by index
    snap = output[100]
    snap = output[-1]  # last snapshot

    # Access data
    radius = snap['R2']
    velocity = snap.get('v2', default=0.0)

    # Properties
    print(snap.t_now)   # Current time [Myr]
    print(snap.phase)   # Current phase name


Data Access Patterns
--------------------

Time Series
^^^^^^^^^^^

Get any parameter across all timesteps:

.. code-block:: python

    # Returns numpy array
    times = output.get('t_now')
    radii = output.get('R2')
    energy = output.get('Eb')

    # For non-numeric data (strings, etc.)
    phases = output.get('current_phase', as_array=False)

Snapshot at Specific Time
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Interpolated (default) - creates synthetic snapshot
    snap = output.get_at_time(1.0)

    # Closest actual snapshot (no interpolation)
    snap = output.get_at_time(1.0, mode='closest')

    # Get just one value
    R2_at_1myr = output.get_at_time(1.0, key='R2')

    # Suppress warning messages
    snap = output.get_at_time(1.0, quiet=True)

Filtering
^^^^^^^^^

.. code-block:: python

    # Filter by simulation phase
    energy_phase = output.filter(phase='energy')
    momentum_phase = output.filter(phase='momentum')

    # Filter by time range
    early = output.filter(t_max=1.0)        # t < 1 Myr
    late = output.filter(t_min=3.0)         # t > 3 Myr
    window = output.filter(t_min=1.0, t_max=2.0)

    # Combine filters
    early_energy = output.filter(phase='energy', t_max=1.0)

    # Result is also a TrinityOutput
    print(len(energy_phase))


Batch Processing Utilities
--------------------------

For parameter sweeps and grid plots, the module provides utilities to find and
organize multiple simulations.

Finding Simulations
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from src._output.trinity_reader import find_all_simulations

    # Find all simulations recursively
    sim_files = find_all_simulations('/path/to/outputs')
    # Returns: List[Path] to dictionary.jsonl files

    for data_path in sim_files:
        output = load_output(data_path)
        # Process each simulation...

Grid Organization
^^^^^^^^^^^^^^^^^

Organize simulations by cloud mass and SFE for grid plots:

.. code-block:: python

    from src._output.trinity_reader import (
        find_all_simulations,
        organize_simulations_for_grid,
        get_unique_ndens
    )

    sim_files = find_all_simulations('/path/to/outputs')

    # Get unique density values
    densities = get_unique_ndens(sim_files)
    # Returns: ['1e3', '1e4'] (sorted)

    # Organize into grid structure
    organized = organize_simulations_for_grid(sim_files, ndens_filter='1e4')

    # Access organized data
    mCloud_list = organized['mCloud_list']  # ['1e5', '1e6', ...]
    sfe_list = organized['sfe_list']        # ['001', '010', ...]
    grid = organized['grid']                 # {(mCloud, sfe): Path}

    # Use in plotting
    for mCloud in mCloud_list:
        for sfe in sfe_list:
            path = grid.get((mCloud, sfe))
            if path:
                output = load_output(path)
                # Plot...


Available Parameters
--------------------

Use ``output.info(verbose=True)`` for complete documentation. Key parameters:

Dynamical Variables
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Unit
     - Description
   * - ``t_now``
     - Myr
     - Current simulation time
   * - ``R2``
     - pc
     - Outer bubble/shell radius
   * - ``v2``
     - pc/Myr
     - Shell expansion velocity
   * - ``Eb``
     - erg
     - Bubble thermal energy
   * - ``T0``
     - K
     - Characteristic bubble temperature
   * - ``R1``
     - pc
     - Inner bubble radius (wind termination shock)
   * - ``Pb``
     - dyn/cm²
     - Bubble pressure

Forces
^^^^^^

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - ``F_grav``
     - Gravitational force
   * - ``F_ram``
     - Ram pressure force (total)
   * - ``F_ram_wind``
     - Ram pressure from stellar winds
   * - ``F_ram_SN``
     - Ram pressure from supernovae
   * - ``F_ion_out``
     - Warm ionized gas force (diagnostic, = P_HII * 4piR2^2, anchored to Pb)
   * - ``F_HII_St``
     - Stroemgren HII force (driving, = P_HII_St * 4piR2^2, independent of Pb)
   * - ``F_rad``
     - Radiation pressure force

Feedback Properties
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Unit
     - Description
   * - ``Lmech_W``
     - erg/Myr
     - Mechanical luminosity from winds
   * - ``Lmech_SN``
     - erg/Myr
     - Mechanical luminosity from supernovae
   * - ``Qi``
     - photons/s
     - Ionizing photon rate
   * - ``Lbol``
     - erg/s
     - Bolometric luminosity


Example: Plotting
-----------------

Basic Radius Evolution
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

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

Phase-Colored Plot
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    t = output.get('t_now')
    R2 = output.get('R2')
    phases = output.get('current_phase', as_array=False)

    colors = {'energy': 'blue', 'momentum': 'red', 'transition': 'orange'}
    c = [colors.get(p, 'gray') for p in phases]

    plt.scatter(t, R2, c=c, s=1)

Force Balance Analysis
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np

    F_grav = np.abs(output.get('F_grav'))
    F_ram = np.abs(output.get('F_ram'))
    F_ion = np.abs(output.get('F_ion_out'))
    F_rad = np.abs(output.get('F_rad'))

    forces = np.vstack([F_grav, F_ram, F_ion, F_rad])
    dominant = np.argmax(forces, axis=0)

    labels = ['Gravity', 'Ram', 'Ionization', 'Radiation']
    for i, label in enumerate(labels):
        frac = np.sum(dominant == i) / len(dominant) * 100
        print(f"{label}: {frac:.1f}%")


File Format Notes
-----------------

For the full specification of TRINITY's output — how ``DescribedDict`` maps
keys to ``DescribedItem`` objects, what each snapshot contains, how the
save/flush pipeline writes ``dictionary.jsonl``, and how to reload snapshots
from Python — see :ref:`sec-running` (*Output Data Model*). This section
covers only details that matter for reader users.

**Legacy format**: the reader automatically handles both ``.jsonl`` (current,
one JSON object per line) and ``.json`` (legacy, pre-2026, a single JSON
object keyed by snapshot id). You do not need to convert old files.

**Snapshot consistency**: all values in a snapshot correspond to the same
timestamp (``t_now``). Snapshots are saved before ODE integration to ensure
consistency across keys, so ``output.get_at_time(t)`` never mixes pre- and
post-step state.

**Profile Array Simplification**: A handful of long 1-D profile arrays are
downsampled before serialisation to keep snapshot size manageable.  Each
simplified array is paired with its own abscissa:

* ``log_bubble_T_arr``     + ``bubble_T_arr_r_arr``     (:math:`\log_{10} T`)
* ``log_bubble_n_arr``     + ``bubble_n_arr_r_arr``     (:math:`\log_{10} n`)
* ``log_bubble_dTdr_arr``  + ``bubble_dTdr_arr_r_arr``  (:math:`\log_{10} |dT/dr|`)
* ``bubble_v_arr``         + ``bubble_v_arr_r_arr``     (velocity, linear)
* ``shell_grav_force_m``   + ``shell_grav_r``           (:math:`\log_{10} |F_{\rm grav}|`)
* ``log_shell_n_arr``      + ``shell_r_arr``            (:math:`\log_{10} n_{\rm shell}`)

The simplifier (``src/_functions/simplify.py``) combines three feature
detectors with a persistence filter and an R²-budgeted thinning step.
Let :math:`\{(x_i, y_i)\}_{i=0}^{n-1}` denote the input curve.

*Menger curvature* is computed for every interior triplet
:math:`(P_{i-1}, P_i, P_{i+1})`:

.. math::

    \kappa_i = \frac{2\,|(P_i - P_{i-1}) \times (P_{i+1} - P_i)|}
                    {\|P_i - P_{i-1}\|\,\|P_{i+1} - P_i\|\,\|P_{i+1} - P_{i-1}\|},

which is the reciprocal of the circumradius of the triplet.  Points with
:math:`\kappa_i > \texttt{grad\_inc}` mark sharp bends.  Sign-change
detection (:math:`\mathrm{sign}(y'_{i+1}) \neq \mathrm{sign}(y'_i)`) adds
every local extremum.

A topological-persistence filter then marks any extremum whose prominence
satisfies

.. math::

    \mathrm{prom}(i) \;\geq\; 0.05 \, \bigl(\max y - \min y\bigr)

as *mandatory* — such points are present at every output budget, so
prominent dips/spikes never flicker in and out across snapshots.

Finally, R²-based thinning picks the smallest subset :math:`S` such that
the linear interpolant :math:`\hat y_S` satisfies

.. math::

    R^2(S) \;=\; 1 - \frac{\sum_i (y_i - \hat y_S(x_i))^2}
                          {\sum_i (y_i - \bar y)^2}
           \;\geq\; 0.99

(the ``r2_target`` argument; default :math:`0.99`).  Candidate subsets
are enumerated in hierarchical-bisection order so that
:math:`S_{N-1} \subset S_N` for every budget :math:`N`, making the output
stable under small changes in ``nmin``.

To recover a profile, linearly interpolate between the paired
``*_r_arr`` abscissa and the (possibly log-space) values.


See Also
--------

- :ref:`sec-running` for simulation execution
- :ref:`sec-visualization` for plotting scripts that use this API
- :ref:`sec-parameters` for parameter definitions
- Full documentation: ``docs/TRINITY_READER.md``
