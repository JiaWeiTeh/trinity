.. highlight:: rest

.. _sec-trinity-reader:

Output Reader API
=================

TRINITY ships a reader module, ``trinity._output.trinity_reader``, with
a small set of routines for loading and manipulating simulation output.
For most users this is the simplest way to access data: it hides the
JSONL layout, the distinction between current ``.jsonl`` and legacy
``.json`` files, the ``metadata.json`` run-constants, and the per-key
unit metadata behind a handful of high-level classes.

Two objects do most of the work. A :class:`TrinityOutput` represents an
entire simulation and exposes time-series access, filtering by phase or
time window, and conversion to a pandas ``DataFrame``. A
:class:`Snapshot` represents a single time step and behaves like a
dictionary keyed by parameter name. A companion utility,
``find_all_simulations``, walks a directory tree of sweep output and
returns the path of every ``dictionary.jsonl`` it finds. The plotting
scripts under ``paper/methods/figures/`` and ``docs/dev/scratch/`` consume their input
exclusively through these classes.

.. seealso::

   - :ref:`sec-running` — *Output data model* — the on-disk
     ``dictionary.jsonl`` and ``metadata.json`` layout.
   - :ref:`sec-parameters` — full list of parameter names and units that
     can appear as keys in a snapshot.
   - :ref:`sec-visualization` — ready-made plotting scripts that consume
     reader output.


Quick start
-----------

.. code-block:: python

    from trinity._output.trinity_reader import load_output

    # Load a simulation
    output = load_output('/path/to/dictionary.jsonl')

    # Basic info
    output.info()

    # Time series as numpy arrays
    times    = output.get('t_now')   # [Myr]
    radii    = output.get('R2')      # [pc]
    velocity = output.get('v2')      # [pc/Myr]

    # Snapshot at a specific time (interpolated)
    snap = output.get_at_time(1.0)
    print(snap['R2'], snap['v2'])

    # Filter by phase
    energy_data = output.filter(phase='energy')

    # Convert to pandas
    df = output.to_dataframe()


Core classes
------------

TrinityOutput
^^^^^^^^^^^^^

The main reader class for TRINITY output files.

.. code-block:: python

    from trinity._output.trinity_reader import TrinityOutput

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
     - Model identifier from the first snapshot
   * - ``keys``
     - List[str]
     - All available parameter names
   * - ``phases``
     - List[str]
     - Unique simulation phases found
   * - ``t_min`` / ``t_max``
     - float
     - Minimum / maximum time in output [Myr]
   * - ``termination`` / ``final_state``
     - dict
     - The ``metadata.json`` end-of-run blocks (see :ref:`sec-running`)

**Methods:**

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Method
     - Description
   * - ``open(filepath)``
     - Class method to open a file
   * - ``get(key, as_array=True)``
     - Parameter across all snapshots as a numpy array (set
       ``as_array=False`` for non-numeric data such as strings)
   * - ``get_at_time(t, mode='interpolate', key=None, quiet=False)``
     - Snapshot at a specific time (``mode='closest'`` for the nearest
       actual snapshot; ``key=`` to return just one value)
   * - ``filter(phase=None, t_min=None, t_max=None)``
     - Filter snapshots by phase and/or time range; returns a
       ``TrinityOutput``
   * - ``info(verbose=False)``
     - Print file information; ``verbose=True`` lists every key with its
       ``info`` string and ``ori_units``
   * - ``to_dataframe()``
     - Convert to a pandas ``DataFrame``

Snapshot
^^^^^^^^

Represents a single simulation timestep.

.. code-block:: python

    snap = output[100]
    snap = output[-1]              # last snapshot

    radius   = snap['R2']
    velocity = snap.get('v2', default=0.0)

    print(snap.t_now)              # current time [Myr]
    print(snap.phase)              # current phase name


Data access patterns
--------------------

Time series
^^^^^^^^^^^

.. code-block:: python

    times  = output.get('t_now')
    radii  = output.get('R2')
    energy = output.get('Eb')

    # Non-numeric data
    phases = output.get('current_phase', as_array=False)

Snapshot at a specific time
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    snap = output.get_at_time(1.0)                 # interpolated (default)
    snap = output.get_at_time(1.0, mode='closest') # nearest actual snapshot
    R2   = output.get_at_time(1.0, key='R2')       # just one value
    snap = output.get_at_time(1.0, quiet=True)     # suppress warnings

Filtering
^^^^^^^^^

.. code-block:: python

    energy_phase   = output.filter(phase='energy')
    momentum_phase = output.filter(phase='momentum')

    early  = output.filter(t_max=1.0)              # t < 1 Myr
    late   = output.filter(t_min=3.0)              # t > 3 Myr
    window = output.filter(t_min=1.0, t_max=2.0)

    early_energy = output.filter(phase='energy', t_max=1.0)
    print(len(energy_phase))                       # result is a TrinityOutput


Batch processing utilities
--------------------------

For parameter sweeps and grid plots, the module finds and organises
multiple simulations.

.. code-block:: python

    from trinity._output.trinity_reader import (
        find_all_simulations,
        organize_simulations_for_grid,
        get_unique_ndens,
        load_output,
    )

    sim_files = find_all_simulations('/path/to/outputs')   # List[Path]

    densities = get_unique_ndens(sim_files)                # e.g. ['1e3', '1e4']

    organized = organize_simulations_for_grid(sim_files, ndens_filter='1e4')
    for mCloud in organized['mCloud_list']:
        for sfe in organized['sfe_list']:
            path = organized['grid'].get((mCloud, sfe))
            if path:
                output = load_output(path)
                # plot ...


Available parameters
--------------------

The full list of parameter names, units, and descriptions that can
appear as snapshot keys lives in :ref:`sec-parameters`. For a quick look
at what is present in a particular file, call
``output.info(verbose=True)``, which prints every key with its attached
``info`` string and ``ori_units``.


File format notes
-----------------

The on-disk ``dictionary.jsonl`` and ``metadata.json`` layout is
described in :ref:`sec-running` (*Output data model*).

**Legacy format**: the reader automatically handles both ``.jsonl``
(current, one JSON object per line) and ``.json`` (legacy, pre-2026, a
single JSON object keyed by snapshot id). You do not need to convert old
files.

**Snapshot consistency**: all values in a snapshot correspond to the
same ``t_now``. Snapshots are saved before ODE integration, so
``output.get_at_time(t)`` never mixes pre- and post-step state.

**Profile arrays**: a handful of long 1-D profile arrays are downsampled
before serialisation (paired ``*_r_arr`` abscissa, log-space where the
values span many decades). The point budget is set by
``simplify_npoints`` (see :ref:`sec-parameters`).
