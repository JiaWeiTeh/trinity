.. highlight:: rest

.. _sec-physics:

Physics & Equations
===================

TRINITY treats the region around a young stellar cluster as a
spherically symmetric two-zone flow. Inside an evacuated cavity, a
hot wind-blown bubble is bounded by an inner termination shock at
:math:`R_1` and by a thin swept-up shell at :math:`R_2`; outside the
shell the ambient cloud retains its initial density profile. The
shell carries essentially all of the swept-up mass, and its motion
is governed by the balance between the thermal and ram pressures of
the interior, the pressure of the photo-ionised gas, direct and
reprocessed radiation pressure, and the gravitational pull of the
cluster and the enclosed cloud mass.

This chapter collects the governing equations, the cloud-initialisation
procedure, the feedback terms drawn from Starburst99, the cooling
prescriptions, and the numerical scheme that integrates the coupled
ODEs. The same equations appear verbatim in the source as the
right-hand side of the integrator, and the symbols used below
correspond one-to-one with the parameter names recorded in the
output (see :ref:`sec-parameters`).


Cloud Initialization
--------------------

Overview
^^^^^^^^

Before evolving the system, TRINITY initializes the molecular cloud structure using ``get_InitCloudProp``. This function:

1. Computes the cloud radius :math:`r_{\rm cloud}` from mass and density constraints
2. Builds radial arrays for density :math:`n(r)` and enclosed mass :math:`M(r)`
3. Validates physical consistency (e.g., edge density > ISM density)

Two density profile types are supported:

- **Power-law (densPL)**: Prescribed analytical profile with exponent :math:`\alpha`
- **Bonnor-Ebert (densBE)**: Self-gravitating isothermal sphere in hydrostatic equilibrium


Power-Law Density Profile
^^^^^^^^^^^^^^^^^^^^^^^^^

The power-law profile follows a piecewise function:

.. math::
   :label: eq-powerlaw-density

   n(r) = \begin{cases}
   n_{\rm core} & r \leq r_{\rm core} \\[0.5em]
   n_{\rm core} \left(\dfrac{r}{r_{\rm core}}\right)^\alpha & r_{\rm core} < r \leq r_{\rm cloud} \\[0.5em]
   n_{\rm ISM} & r > r_{\rm cloud}
   \end{cases}

where :math:`n_{\rm core}` is the core number density (``nCore``),
:math:`r_{\rm core}` the core radius (``rCore``), :math:`\alpha` the
power-law exponent (``densPL_alpha``), and :math:`n_{\rm ISM}` the
ambient ISM density (``nISM``).

The enclosed mass :math:`M(r)` is obtained by integrating this
profile, following `Rahner et al. (2018)
<https://ui.adsabs.harvard.edu/abs/2018MNRAS.473.4862R>`_ Eq. 25.
For homogeneous clouds (:math:`\alpha = 0`):

.. math::
   :label: eq-mass-homogeneous

   M(r) = \frac{4}{3}\pi r^3 \rho_{\rm core},

with :math:`\rho_{\rm core} = \mu \, m_{\rm H} \, n_{\rm core}`.
For :math:`\alpha \neq 0` and :math:`r_{\rm core} < r \leq r_{\rm cloud}`:

.. math::
   :label: eq-mass-powerlaw

   M(r) = 4\pi\rho_{\rm core} \left[ \frac{r_{\rm core}^3}{3} + \frac{r^{3+\alpha} - r_{\rm core}^{3+\alpha}}{(3+\alpha) \, r_{\rm core}^\alpha} \right].

The cloud radius :math:`r_{\rm cloud}` is then determined by solving
:math:`M(r_{\rm cloud}) = M_{\rm cloud}` via Brent's method.

The exponent :math:`\alpha` controls the density gradient (valid
range :math:`-2 \leq \alpha \leq 0`):

.. list-table::
   :widths: 15 25 60
   :header-rows: 1

   * - :math:`\alpha`
     - Profile Type
     - Physical Interpretation
   * - 0
     - Homogeneous
     - Uniform density throughout. Simplest case.
   * - -1
     - Intermediate
     - Moderate density gradient.
   * - -1.5
     - Typical GMC
     - Observed in many giant molecular clouds.
   * - -2
     - Isothermal sphere
     - Singular isothermal sphere :math:`\rho \propto r^{-2}`. Self-similar collapse solution.


Bonnor-Ebert Sphere
^^^^^^^^^^^^^^^^^^^

A Bonnor-Ebert sphere is an isothermal, self-gravitating gas sphere
in hydrostatic equilibrium, confined by external pressure — relevant
for molecular cloud cores near the threshold of gravitational
collapse. It is determined by three properties: isothermality
(constant temperature/sound speed), self-gravity (density follows
from pressure–gravity balance), and pressure confinement (external
pressure truncates the sphere at :math:`r_{\rm cloud}`).

The density structure is governed by the isothermal Lane-Emden
equation:

.. math::
   :label: eq-lane-emden

   \frac{d^2 u}{d\xi^2} + \frac{2}{\xi}\frac{du}{d\xi} = e^{-u}

where :math:`\xi = r/a` is the dimensionless radius,
:math:`u(\xi)` the dimensionless gravitational potential, and
:math:`a = c_s / \sqrt{4\pi G \rho_c}` the scale radius. Boundary
conditions at the centre are :math:`u(0) = 0,\ du/d\xi|_{\xi=0} = 0`,
and the density contrast relative to the centre is
:math:`\rho(\xi)/\rho_c = e^{-u(\xi)}`.

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Symbol
     - Parameter
     - Description
   * - :math:`\Omega`
     - ``densBE_Omega``
     - Density contrast :math:`\rho_{\rm core}/\rho_{\rm edge}`. Controls how centrally concentrated the sphere is.
   * - :math:`\xi_{\rm out}`
     - (computed)
     - Dimensionless outer radius where density equals :math:`\rho_c/\Omega`.
   * - :math:`c_s`
     - (computed)
     - Isothermal sound speed, determined self-consistently from mass constraint.

The dimensionless mass from the Lane-Emden solution is
:math:`m(\xi) = -\xi^2\,du/d\xi`, which converts to physical mass via
:math:`M = m(\xi)\, \rho_c\, a^3`. The critical density contrast is
:math:`\Omega_{\rm crit} \approx 14.04`, corresponding to
:math:`\xi_{\rm crit} \approx 6.45`.

.. warning::

   Spheres with :math:`\Omega > \Omega_{\rm crit}` are
   gravitationally unstable and will collapse. TRINITY allows such
   configurations but the user should be aware of the physical
   implications.

Unlike the power-law profile, the Bonnor-Ebert density is smooth
throughout the cloud and has no flat core region; it falls
monotonically from :math:`n_c` at the centre to the edge density at
:math:`r_{\rm cloud}`, beyond which the medium is taken to have
constant density :math:`n_{\rm ISM}`.


Mass Accretion Rate
^^^^^^^^^^^^^^^^^^^

As the shell expands through the cloud, it sweeps up mass. The instantaneous mass accretion rate is:

.. math::
   :label: eq-mdot

   \frac{dM_{\rm shell}}{dt} = 4\pi r^2 \rho(r) \, v(r)

where:

- :math:`r` is the current shell radius
- :math:`\rho(r)` is the local density from the cloud profile
- :math:`v(r) = dr/dt` is the shell velocity

This is the mass flux through a spherical surface moving at velocity :math:`v`.

.. note::

   This formula applies universally to both power-law and Bonnor-Ebert profiles. The profile shape only affects :math:`\rho(r)`.


Shell Dynamics
--------------

Momentum Equation
^^^^^^^^^^^^^^^^^

The shell momentum equation governs the expansion dynamics:

.. math::
   :label: eq-momentum

   \frac{d}{dt}(M_{\rm sh} v) = 4\pi R^2 (P_{\rm drive} - P_{\rm ext}) + F_{\rm rad} - F_{\rm grav}

where:

- :math:`M_{\rm sh}` is the shell mass
- :math:`v = dR/dt` is the shell velocity
- :math:`R` is the shell radius
- :math:`P_{\rm drive}` is the driving pressure
- :math:`P_{\rm ext}` is the external confining pressure
- :math:`F_{\rm rad}` is the radiation pressure force
- :math:`F_{\rm grav}` is the gravitational force

Expanding the left-hand side gives the acceleration:

.. math::

   M_{\rm sh} \dot{v} = F_{\rm gas} + F_{\rm rad} - F_{\rm grav} - \dot{M}_{\rm sh} v

The mass-loading term :math:`\dot{M}_{\rm sh} v` represents momentum loss as new material is swept into the shell.


Driving Pressure Model
^^^^^^^^^^^^^^^^^^^^^^

TRINITY uses a **convex blend model** for the driving pressure that smoothly transitions
between energy-driven (hot bubble) and momentum-driven (HII region) regimes:

.. math::
   :label: eq-pdrive

   P_{\rm drive} = (1 - w) P_b + w P_{\rm IF}

where:

- :math:`P_b` is the hot bubble thermal pressure
- :math:`P_{\rm IF}` is the ionization front pressure
- :math:`w` is the blending weight

The blending weight is determined by:

.. math::
   :label: eq-wblend

   w = f_{\rm abs,ion} \cdot \frac{P_{\rm IF}}{P_{\rm IF} + P_b}

where :math:`f_{\rm abs,ion}` is the fraction of ionizing photons absorbed in the shell.

**Physical interpretation:**

- **Early times** (:math:`w \approx 0`): Hot bubble pressure dominates (energy-driven expansion)
- **Late times** (:math:`w \approx 1`): Warm ionized gas pressure dominates (HII-driven expansion)
- **Transition**: Smooth handoff as bubble cools or leaks

The ionization front pressure is computed from the shell structure:

.. math::

   P_{\rm IF} = 2 n_{\rm IF} k_B T_{\rm ion}

where :math:`n_{\rm IF}` is the density at the ionization front and :math:`T_{\rm ion} \approx 10^4` K
is the ionized gas temperature.


Force Components
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Force
     - Expression
     - Physical Origin
   * - :math:`F_{\rm gas}`
     - :math:`4\pi R^2 (P_{\rm drive} - P_{\rm ext})`
     - Net thermal pressure (inward + outward)
   * - :math:`F_{\rm rad}`
     - :math:`\frac{L_{\rm abs}}{c}(1 + \tau_{\rm IR})`
     - Direct + reprocessed radiation momentum
   * - :math:`F_{\rm grav}`
     - :math:`\frac{G M_{\rm sh}(M_* + M_{\rm sh}/2)}{R^2}`
     - Self-gravity of shell + cluster
   * - :math:`F_{\rm ram}`
     - (energy phase only)
     - Hot bubble ram pressure


Phase Evolution
^^^^^^^^^^^^^^^

The driving-pressure model described above is evaluated in a
specific sequence of dynamical regimes — energy-driven,
transition, and momentum-driven — that are implemented as
separate solvers with their own exit criteria. The per-phase
solver definitions, exit conditions, and orchestrator flow are
documented in :ref:`sec-architecture` (*Simulation Phases*).
