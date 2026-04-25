.. highlight:: rest

.. _sec-architecture:

Physics Architecture
====================

Internally, TRINITY is organised as a small orchestrator that drives
a sequence of phase-specific solvers, each of which consumes the
same set of shared physics modules. A single state dictionary,
built on the ``DescribedDict`` container described in
:ref:`sec-running`, is threaded through every call and is the sole
mechanism by which the modules exchange information. The
architecture is deliberately flat: there are no class hierarchies,
no dependency injection, and no plugin system. Physics modules are
plain functions that read and write named keys on the state
dictionary, and the feedback loop that advances the shell in time
is a single ODE right-hand side whose inputs and outputs are
traceable directly to the equations in :ref:`sec-physics`.


Module Organisation
-------------------

The source tree under ``src/`` is split into four layers:

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
  ``cloud_properties/``, ``sb99/``, ``cooling/``,
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
   \frac{dE_b}{dt} &= L_{\rm gain} - L_{\rm loss} - P_b \frac{dV}{dt}, \\
   \frac{dT_0}{dt} &= \frac{T_0}{t}\,\delta.

The updated state is written back to the dictionary, a snapshot is
staged if the save interval has elapsed (see :ref:`sec-running`,
*Output Data Model*), and control returns to the orchestrator for
the next step.


See Also
--------

- :ref:`sec-running` — how to execute simulations.
- :ref:`sec-parameters` — parameter names, units, and defaults.
- :ref:`sec-physics` — derivation of the equations integrated above.
- :ref:`sec-trinity-reader` — API for reading and analysing output.
