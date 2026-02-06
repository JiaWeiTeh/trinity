.. highlight:: rest

.. _sec-architecture:

Physics Architecture
====================

This section documents the internal architecture of TRINITY, illustrating how the simulation modules interact to model HII region evolution. The diagrams below show module dependencies, simulation phases, data flow, and the core physics feedback loop.


Module Organization
-------------------

TRINITY is organized into layered modules: an orchestrator, phase-specific solvers, shared physics modules, and utilities.

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────┐
    │                          ORCHESTRATOR                               │
    │                            main.py                                  │
    │              start_expansion() ───► run_expansion()                 │
    └──────────────────────────────┬──────────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
    ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
    │ phase0_init │        │    sb99/    │        │   cooling/  │
    │             │        │             │        │             │
    │ Cloud props │◄───────│  read_SB99  │        │ CIE curves  │
    │ Init state  │        │  feedback   │        │ non-CIE     │
    └──────┬──────┘        └──────┬──────┘        └──────┬──────┘
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        PHASE MODULES                                │
    │                                                                     │
    │  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐  │
    │  │  phase1    │──►│  phase1b   │──►│  phase1c   │──►│  phase2    │  │
    │  │  _energy   │   │  _implicit │   │  _transit  │   │  _momentum │  │
    │  │            │   │            │   │            │   │            │  │
    │  │ Constant   │   │ Adaptive   │   │ Energy ─►  │   │ Pure       │  │
    │  │ cooling    │   │ cooling    │   │ Momentum   │   │ dynamics   │  │
    │  └────────────┘   └────────────┘   └────────────┘   └────────────┘  │
    └──────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      SHARED PHYSICS MODULES                         │
    │                                                                     │
    │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │
    │  │ phase_general/ │  │bubble_structure│  │   shell_structure/     │ │
    │  │                │  │                │  │                        │ │
    │  │  phase_ODEs    │  │ get_bubbleParam│  │  shell_structure       │ │
    │  │  (dv/dt, dE/dt)│  │ bubble_luminosi│  │  get_shellParams       │ │
    │  └────────────────┘  └────────────────┘  └────────────────────────┘ │
    │                                                                     │
    │  ┌────────────────────────────┐  ┌────────────────────────────────┐ │
    │  │    cloud_properties/       │  │         _functions/            │ │
    │  │                            │  │                                │ │
    │  │  density_profile           │  │  unit_conversions              │ │
    │  │  mass_profile              │  │  operations                    │ │
    │  │  powerLawSphere            │  │  logging_setup                 │ │
    │  │  bonnorEbertSphere         │  │                                │ │
    │  └────────────────────────────┘  └────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        UTILITY MODULES                              │
    │                                                                     │
    │  ┌──────────────────┐              ┌──────────────────┐             │
    │  │     _input/      │              │     _output/     │             │
    │  │                  │              │                  │             │
    │  │  read_param      │              │  trinity_reader  │             │
    │  │  dictionary      │              │  simulation_end  │             │
    │  │  (DescribedDict) │              │  terminal_prints │             │
    │  └──────────────────┘              └──────────────────┘             │
    └─────────────────────────────────────────────────────────────────────┘


Simulation Phases
-----------------

The simulation progresses through distinct phases, each with different physics assumptions.

.. code-block:: text

    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                      PHASE 0: INITIALIZATION                          ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  1. Read parameter file ──► DescribedDict params                      ║
    ║  2. Build cloud density profile n(r) and mass profile M(r)            ║
    ║  3. Load SB99 tables ──► interpolation functions fLmech(t), fQi(t)    ║
    ║  4. Load cooling curves (CIE + non-CIE)                               ║
    ║  5. Calculate initial state from free-streaming ──► Weaver solution   ║
    ║                                                                       ║
    ║  Output: y₀ = [R2₀, v2₀, Eb₀, T0₀]                                    ║
    ╚═══════════════════════════════════╤═══════════════════════════════════╝
                                        │
                                        ▼
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║              PHASE 1a: ENERGY-DRIVEN (Constant Cooling)               ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  Physics: Weaver+77 wind-blown bubble with radiative cooling          ║
    ║                                                                       ║
    ║  • Bubble pressure Pb drives shell expansion                          ║
    ║  • Cooling parameters (α, β, δ) held constant                         ║
    ║  • Shell structure computed at each timestep                          ║
    ║  • Gravity from cluster + swept-up mass                               ║
    ║                                                                       ║
    ║  Exit when: R2 approaches rCloud OR cooling becomes significant       ║
    ╚═══════════════════════════════════╤═══════════════════════════════════╝
                                        │
                                        ▼
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║              PHASE 1b: ENERGY-DRIVEN (Adaptive Cooling)               ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  Physics: Implicit cooling integration                                ║
    ║                                                                       ║
    ║  • α = v2 · t / R2  (expansion parameter)                             ║
    ║  • β = -dPb/dt      (pressure evolution)                              ║
    ║  • δ = dT/dt        (temperature evolution)                           ║
    ║                                                                       ║
    ║  Monitors: Lgain vs Lloss (energy balance)                            ║
    ║                                                                       ║
    ║  Exit when: (Lgain - Lloss)/Lgain < threshold                         ║
    ╚═══════════════════════════════════╤═══════════════════════════════════╝
                                        │
                                        ▼
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                      PHASE 1c: TRANSITION                             ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  Physics: Energy dissipation bridge                                   ║
    ║                                                                       ║
    ║  • dEb/dt = -Eb / t_soundcrossing                                     ║
    ║  • Bubble energy Eb ──► 0                                             ║
    ║  • Shell dynamics continue                                            ║
    ║                                                                       ║
    ║  Exit when: Eb < threshold (≈ 0)                                      ║
    ╚═══════════════════════════════════╤═══════════════════════════════════╝
                                        │
                                        ▼
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                    PHASE 2: MOMENTUM-DRIVEN                           ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  Physics: No thermal pressure (Eb = 0)                                ║
    ║                                                                       ║
    ║  Shell driven by:                                                     ║
    ║    • Ram pressure from winds: Pram = Lmech / (2π R² vmech)            ║
    ║    • Radiation pressure: Frad = fabs · Lbol / c                       ║
    ║    • Gravity: Fgrav = G · Msh · (Mcluster + Msh/2) / R²               ║
    ║                                                                       ║
    ║  Exit when: stop_t OR stop_r OR shell dissolved OR collapse           ║
    ╚═══════════════════════════════════════════════════════════════════════╝


State Variables
---------------

The simulation evolves a state vector through the ODE system:

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
     - Outer bubble/shell radius
   * - ``v2``
     - :math:`v_2`
     - pc/Myr
     - Shell radial velocity
   * - ``Eb``
     - :math:`E_b`
     - AU
     - Bubble thermal energy (internal units)
   * - ``T0``
     - :math:`T_0`
     - K
     - Bubble central temperature


Data Flow
---------

Data transforms through the simulation as follows:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        STATE VECTOR                                     │
    │                                                                         │
    │              y = [ R2,     v2,      Eb,      T0    ]                     │
    │                    │       │        │        │                          │
    │                  Shell   Shell   Bubble   Bubble                        │
    │                  Radius  Velocity Energy   Temp                         │
    └────────────────────┼───────┼────────┼────────┼──────────────────────────┘
                         │       │        │        │
                         ▼       ▼        ▼        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     SB99 INTERPOLATION                                  │
    │                                                                         │
    │   t_now ────┬────► Qi(t)     ─────────────► Ionizing photon rate        │
    │             │                                                           │
    │             ├────► Lbol(t)   ─────────────► Bolometric luminosity       │
    │             │                                                           │
    │             ├────► Lmech(t)  ─────────────► Mechanical luminosity       │
    │             │                  (wind + SN)                              │
    │             │                                                           │
    │             └────► pdot(t)   ─────────────► Momentum injection rate     │
    └──────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     BUBBLE STRUCTURE                                    │
    │                                                                         │
    │   Inputs: R2, Eb, Lmech, vWind                                          │
    │                                                                         │
    │   ┌───────────────────────────────────────────────────────────────┐     │
    │   │  R1 = solve( pressure balance at inner discontinuity )        │     │
    │   │                                                               │     │
    │   │  Pb = (γ-1) · Eb / [ 4π/3 · (R2³ - R1³) ]                     │     │
    │   │                                                               │     │
    │   │  T(r), n(r), v(r) profiles via Weaver+77 Eqs 42-43            │     │
    │   └───────────────────────────────────────────────────────────────┘     │
    │                                                                         │
    │   Outputs: R1, Pb, Lgain, Lloss, bubble profiles                        │
    └──────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      SHELL STRUCTURE                                    │
    │                                                                         │
    │   Inputs: Pb, R2, Mshell, Qi, Li, Ln                                    │
    │                                                                         │
    │   ┌───────────────────────────────────────────────────────────────┐     │
    │   │  nShell₀ = Pb / (kB · Tion)  (pressure equilibrium)           │     │
    │   │                                                               │     │
    │   │  Ionized region: ODE for n(r), τ(r) until Qi absorbed         │     │
    │   │                                                               │     │
    │   │  Neutral region: density jump, continue to shell edge         │     │
    │   └───────────────────────────────────────────────────────────────┘     │
    │                                                                         │
    │   Outputs: fabs_ion, fabs_neu, nMax, shell_thickness, gravity           │
    └──────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        ODE SOLVER                                       │
    │                                                                         │
    │   ┌───────────────────────────────────────────────────────────────┐     │
    │   │  dR2/dt = v2                                                  │     │
    │   │                                                               │     │
    │   │  dv2/dt = [ 4πR2²·Pb - Fgrav + Frad - (dMsh/dt)·v2 ] / Msh    │     │
    │   │                                                               │     │
    │   │  dEb/dt = Lgain - Lloss - Pb·dV/dt                            │     │
    │   │                                                               │     │
    │   │  dT0/dt = (T0/t) · δ                                          │     │
    │   └───────────────────────────────────────────────────────────────┘     │
    │                                                                         │
    │   Output: dy/dt ──► integrate ──► y(t + dt)                             │
    └─────────────────────────────────────────────────────────────────────────┘


Physics Feedback Loop
---------------------

The core of TRINITY is a self-consistent feedback loop coupling stellar feedback, bubble dynamics, shell structure, and gravity:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │   ╔═══════════════════╗                                                 │
    │   ║  STELLAR FEEDBACK ║◄──────────────────────────────────────────┐     │
    │   ║     (SB99)        ║                                           │     │
    │   ╚═════════╤═════════╝                                           │     │
    │             │                                                     │     │
    │             │  Interpolate at t_now:                              │     │
    │             │    • Lmech (mechanical luminosity)                  │     │
    │             │    • pdot  (momentum rate)                          │     │
    │             │    • Qi    (ionizing photons)                       │     │
    │             │    • Lbol  (bolometric luminosity)                  │     │
    │             ▼                                                     │     │
    │   ╔═══════════════════╗                                           │     │
    │   ║ BUBBLE STRUCTURE  ║                                           │     │
    │   ╚═════════╤═════════╝                                           │     │
    │             │                                                     │     │
    │             │  Calculate:                                         │     │
    │             │    • R1 (inner shock radius)                        │     │
    │             │    • Pb (bubble pressure)                           │     │
    │             │    • T(r), n(r) profiles                            │     │
    │             │    • Lgain, Lloss (cooling)                         │     │
    │             ▼                                                     │     │
    │   ╔═══════════════════╗                                           │     │
    │   ║  SHELL STRUCTURE  ║                                           │     │
    │   ╚═════════╤═════════╝                                           │     │
    │             │                                                     │     │
    │             │  Calculate:                                         │     │
    │             │    • Shell density profile                          │     │
    │             │    • Radiation absorption                           │     │
    │             │    • Ionization state                               │     │
    │             ▼                                                     │     │
    │   ╔═══════════════════╗                                           │     │
    │   ║      GRAVITY      ║                                           │     │
    │   ╚═════════╤═════════╝                                           │     │
    │             │                                                     │     │
    │             │  Calculate:                                         │     │
    │             │    • Mshell from cloud profile                      │     │
    │             │    • Fgrav = G·M·(Mcluster + Msh/2)/R²              │     │
    │             ▼                                                     │     │
    │   ╔═══════════════════╗                                           │     │
    │   ║   ODE DYNAMICS    ║                                           │     │
    │   ╚═════════╤═════════╝                                           │     │
    │             │                                                     │     │
    │             │  Solve: dR2/dt, dv2/dt, dEb/dt, dT0/dt              │     │
    │             ▼                                                     │     │
    │   ╔═══════════════════╗                                           │     │
    │   ║   STATE UPDATE    ║                                           │     │
    │   ╚═════════╤═════════╝                                           │     │
    │             │                                                     │     │
    │             │  y = y + dy·dt                                      │     │
    │             │  t_now = t_now + dt                                 │     │
    │             │                                                     │     │
    │             └─────────────────────────────────────────────────────┘     │
    │                                                                         │
    │                            NEXT TIMESTEP                                │
    └─────────────────────────────────────────────────────────────────────────┘


See Also
--------

- :ref:`sec-running` for how to execute simulations
- :ref:`sec-parameters` for parameter specifications and units
- :ref:`sec-trinity-reader` for reading and analyzing output data
