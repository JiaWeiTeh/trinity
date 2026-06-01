.. highlight:: rest

.. _sec-architecture:

Physics Architecture
====================

Internally, TRINITY is organised as a small orchestrator that drives
a sequence of phase-specific solvers, each of which consumes the
same set of shared physics modules. A single state dictionary,
built on the ``DescribedDict`` container documented under
*Snapshot Persistence* below, is threaded through every call and is
the sole mechanism by which the modules exchange information. The
architecture is deliberately flat: there are no class hierarchies,
no dependency injection, and no plugin system. Physics modules are
plain functions that read and write named keys on the state
dictionary, and the feedback loop that advances the shell in time
is a single ODE right-hand side whose inputs and outputs are
traceable directly to the equations in :ref:`sec-physics`.


Module Organisation
-------------------

The source tree under ``trinity/`` is split into four layers:

* **Orchestrator** (``main.py``). Entry point
  ``start_expansion()`` / ``run_expansion()`` that advances the
  state dictionary from phase to phase until the stopping
  criterion is reached.
* **Phase solvers** (``phase0_init/``, ``phase1_energy/``,
  ``phase1b_implicit/``, ``phase1c_transit/``,
  ``phase2_momentum/``). One solver per dynamical regime; each
  owns its own ODE right-hand side and its own exit condition.
* **Shared physics modules** (``phase_general/``,
  ``bubble_structure/``, ``shell_structure/``,
  ``cloud_properties/``, ``sps/``, ``cooling/``,
  ``_functions/``). Pure functions that compute bubble
  structure, shell structure, cloud density and mass profiles,
  stellar feedback from Starburst99 tables, cooling functions,
  and unit conversions. Any phase solver may call any of these.
* **I/O utilities** (``_input/``, ``_output/``). Parameter-file
  parsing, the ``DescribedDict`` container, snapshot writing,
  and the reader API of :ref:`sec-trinity-reader`.


Simulation Phases
-----------------

A simulation progresses through up to five phases. Each phase has
its own physics assumptions and its own exit criterion; the
orchestrator hands the state dictionary from one phase to the next
until a terminal condition is met.

* **Phase 0 — Initialisation.** Reads the parameter file, builds
  the cloud density and mass profiles :math:`n(r)` and
  :math:`M(r)`, loads the Starburst99 interpolation tables and the
  cooling curves, and computes the initial free-streaming
  (Weaver) solution used as :math:`y_0`.
* **Phase 1a — Energy-driven, constant cooling.** Weaver+77
  wind-blown bubble with the cooling parameters
  :math:`(\alpha, \beta, \delta)` held constant. The bubble
  pressure :math:`P_b` drives the shell; exit is triggered when
  :math:`R_2` approaches :math:`r_{\rm cloud}` or when radiative
  cooling becomes non-negligible.
* **Phase 1b — Energy-driven, adaptive cooling.** Implicit
  integration of the cooling; :math:`(\alpha, \beta, \delta)`
  are updated at every step. The phase exits when the energy
  balance :math:`(L_{\rm gain} - L_{\rm loss})/L_{\rm gain}`
  falls below a threshold.
* **Phase 1c — Transition.** Energy-dissipation bridge governed
  by :math:`dE_b/dt = -E_b / t_{\rm sc}`. Shell dynamics continue
  to be integrated. Exits when :math:`E_b` approaches zero.
* **Phase 2 — Momentum-driven.** The bubble thermal energy has
  vanished and the shell is driven by ram pressure
  :math:`P_{\rm ram} = L_{\rm mech}/(2\pi R^2 v_{\rm mech})`,
  radiation pressure :math:`F_{\rm rad} = f_{\rm abs} L_{\rm bol}/c`,
  and gravity. The simulation terminates when the user-specified
  stopping time, radius, dissolution criterion, or collapse
  condition is reached.


State Variables
---------------

The simulation evolves a four-component state vector through the
ODE system:

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Variable
     - Symbol
     - Unit
     - Description
   * - ``R2``
     - :math:`R_2`
     - pc
     - Outer bubble radius (= inner shell edge)
   * - ``v2``
     - :math:`v_2`
     - pc/Myr
     - Radial velocity at R2 (outer bubble / inner shell edge)
   * - ``Eb``
     - :math:`E_b`
     - AU (internal)
     - Bubble thermal energy
   * - ``T0``
     - :math:`T_0`
     - K
     - Characteristic bubble temperature

Every other recorded quantity — pressures, forces, luminosities,
profiles — is derived from this vector together with the shared
physics modules.


Feedback Loop
-------------

At each ODE step the right-hand side performs the following
computation. Given the current state :math:`y = (R_2, v_2, E_b, T_0)`
and time :math:`t`, Starburst99 interpolation tables are evaluated
to obtain the mechanical luminosity :math:`L_{\rm mech}`, the
momentum injection rate :math:`\dot p`, the ionising photon rate
:math:`Q_i`, and the bolometric luminosity :math:`L_{\rm bol}`.
The bubble-structure module then solves for the inner shock radius
:math:`R_1`, the bubble pressure :math:`P_b`, and the cooling
gain/loss terms. The shell-structure module uses :math:`P_b` and
the feedback rates to compute the shell density profile, the
fraction of ionising and bolometric radiation absorbed in the
shell, and the gravitational contribution of the swept-up mass.
Finally, the ODE solver advances :math:`y` through

.. math::

   \frac{dR_2}{dt} &= v_2, \\
   \frac{dv_2}{dt} &= \frac{4\pi R_2^2 P_b - F_{\rm grav} + F_{\rm rad} - \dot M_{\rm sh} v_2}{M_{\rm sh}}, \\
   \frac{dE_b}{dt} &= L_{\rm gain} - L_{\rm loss} - L_{\rm leak} - P_b \frac{dV}{dt}, \\
   \frac{dT_0}{dt} &= \frac{T_0}{t}\,\delta.

Here :math:`L_{\rm leak}` is the geometry-set covering-fraction leak, the
enthalpy flux of hot gas venting through the open fraction
:math:`(1-C_{\rm f})` of the bubble wall. It is zero by default
(:math:`C_{\rm f}=1`, sealed bubble) and active only when
``coverFraction`` :math:`< 1`; see the parameter of that name.

The updated state is written back to the dictionary, a snapshot is
staged if the save interval has elapsed (see *Snapshot Persistence*
below), and control returns to the orchestrator for the next step.


Snapshot Persistence
--------------------

Simulation state lives in a single ``DescribedDict`` (defined in
``trinity/_input/dictionary.py``). Each key maps to a ``DescribedItem``
that wraps the raw value together with two pieces of metadata:
``info`` (a short human-readable description) and ``ori_units``
(the original-unit label, e.g. ``"pc"``, ``"Msun"``,
``"1/cm**3"``). A per-item ``exclude_from_snapshot`` flag marks
keys that are not persisted to disk — used for large auxiliary
objects such as SPS interpolation tables that can be rebuilt on
load.

Snapshots are captured through a two-stage *buffer → flush*
pipeline so that disk writes stay cheap (append-only, O(1) per
flush) and a crash can lose at most ``snapshot_interval`` steps of
progress. The sequence at each ODE step is:

1. **Mutate the dict.** Physics modules update
   ``params["R2"].value``, ``params["Eb"].value``, etc. in place.
2. **Stage a snapshot.** ``params.save_snapshot()`` copies the
   current state (excluding any key marked
   ``exclude_from_snapshot=True``) into the in-memory buffer
   ``params.previous_snapshot``. A duplicate guard compares
   ``t_now`` + ``R2`` against the last saved entry and silently
   drops re-runs of the same step.
3. **Flush in batches.** Every ``snapshot_interval`` calls (default
   **10**), ``save_snapshot`` triggers ``flush()`` automatically.
   ``params.flush()`` may also be called manually at phase
   boundaries or after a critical event.
4. **Append to disk.** ``flush()`` opens ``dictionary.jsonl`` in
   append mode and writes one JSON line per pending snapshot, using
   ``NpEncoder`` to serialise numpy scalars and arrays. The first
   flush of a fresh run overwrites any existing file; subsequent
   flushes only append.
5. **Crash-safe handlers.** On construction, ``DescribedDict``
   registers an ``atexit`` hook plus ``SIGINT`` / ``SIGTERM``
   handlers, so that an exit — clean, via ``Ctrl+C``, or via
   ``kill`` / SLURM ``scancel`` — flushes any buffered snapshots
   before termination. ``SIGKILL`` (``kill -9``) and ``os._exit()``
   bypass these hooks and can lose the pending buffer; everything
   already on disk is always safe.

Only the ``.value`` of each ``DescribedItem`` is written to disk —
``info`` and ``ori_units`` live alongside the code and are
reattached automatically when a snapshot is loaded back in.


Profile Array Downsampling
--------------------------

A handful of long 1-D profile arrays are downsampled before
serialisation to keep snapshot size manageable. Each simplified array
is paired with its own abscissa:

* ``log_bubble_T_arr``     + ``bubble_T_arr_r_arr``     (:math:`\log_{10} T`)
* ``log_bubble_n_arr``     + ``bubble_n_arr_r_arr``     (:math:`\log_{10} n`)
* ``log_bubble_dTdr_arr``  + ``bubble_dTdr_arr_r_arr``  (:math:`\log_{10} |dT/dr|`)
* ``bubble_v_arr``         + ``bubble_v_arr_r_arr``     (velocity, linear)
* ``shell_grav_force_m``   + ``shell_grav_r``           (:math:`\log_{10} |F_{\rm grav}|`)
* ``log_shell_n_arr``      + ``shell_r_arr``            (:math:`\log_{10} n_{\rm shell}`)

The simplifier (``trinity/_functions/simplify.py``) combines three feature
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
