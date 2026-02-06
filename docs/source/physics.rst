.. highlight:: rest

.. _sec-physics:

Physics & Equations
===================

This section documents the physics and mathematical formulations implemented in TRINITY. The simulation models HII region evolution using 1D spherical hydrodynamics, coupling stellar feedback with shell dynamics.


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

Density Distribution
""""""""""""""""""""

The power-law profile follows a piecewise function:

.. math::
   :label: eq-powerlaw-density

   n(r) = \begin{cases}
   n_{\rm core} & r \leq r_{\rm core} \\[0.5em]
   n_{\rm core} \left(\dfrac{r}{r_{\rm core}}\right)^\alpha & r_{\rm core} < r \leq r_{\rm cloud} \\[0.5em]
   n_{\rm ISM} & r > r_{\rm cloud}
   \end{cases}

where:

- :math:`n_{\rm core}` is the core number density (``nCore`` parameter)
- :math:`r_{\rm core}` is the core radius (``rCore`` parameter)
- :math:`\alpha` is the power-law exponent (``densPL_alpha`` parameter)
- :math:`n_{\rm ISM}` is the ambient ISM density (``nISM`` parameter)

.. code-block:: text

   Density Profile Shape (α < 0)

   n(r)
    │
    │▓▓▓▓▓▓▓▓▓▓▓▓
    │            ▓▓▓▓
    │                ▓▓▓▓
    │                    ▓▓▓▓
    │                        ▓▓▓▓────────────
    │                             nISM
    └────────────────────────────────────────► r
              rCore    rCloud


Mass Profile
""""""""""""

The enclosed mass :math:`M(r)` is obtained by integrating the density profile. Following `Rahner et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018MNRAS.473.4862R>`_ Eq. 25:

**For homogeneous clouds** (:math:`\alpha = 0`):

.. math::
   :label: eq-mass-homogeneous

   M(r) = \frac{4}{3}\pi r^3 \rho_{\rm core}

where :math:`\rho_{\rm core} = \mu \, m_{\rm H} \, n_{\rm core}` is the mass density.

**For power-law clouds** (:math:`\alpha \neq 0`):

.. math::
   :label: eq-mass-powerlaw

   M(r) = 4\pi\rho_{\rm core} \left[ \frac{r_{\rm core}^3}{3} + \frac{r^{3+\alpha} - r_{\rm core}^{3+\alpha}}{(3+\alpha) \, r_{\rm core}^\alpha} \right]

for :math:`r_{\rm core} < r \leq r_{\rm cloud}`.

The cloud radius :math:`r_{\rm cloud}` is determined by solving :math:`M(r_{\rm cloud}) = M_{\rm cloud}` using root-finding (Brent's method).


Parameter Interpretation
""""""""""""""""""""""""

The power-law exponent :math:`\alpha` determines the density gradient:

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

.. note::

   The parameter ``densPL_alpha`` in the input file corresponds to :math:`\alpha`. Valid range: :math:`-2 \leq \alpha \leq 0`. See :ref:`sec-parameters` for details.


Bonnor-Ebert Sphere
^^^^^^^^^^^^^^^^^^^

Physical Background
"""""""""""""""""""

A Bonnor-Ebert sphere is an isothermal, self-gravitating gas sphere in hydrostatic equilibrium, confined by external pressure. It represents a more physically motivated initial condition than a prescribed power-law.

The key physics:

- **Isothermal**: Constant temperature (and sound speed) throughout
- **Self-gravitating**: Density profile determined by balance of pressure and gravity
- **Pressure-confined**: External pressure truncates the sphere at :math:`r_{\rm cloud}`

This configuration is relevant for molecular cloud cores near the threshold of gravitational collapse.


Lane-Emden Equation
"""""""""""""""""""

The density structure is governed by the **isothermal Lane-Emden equation**:

.. math::
   :label: eq-lane-emden

   \frac{d^2 u}{d\xi^2} + \frac{2}{\xi}\frac{du}{d\xi} = e^{-u}

where:

- :math:`\xi = r/a` is the dimensionless radius
- :math:`u(\xi)` is the dimensionless gravitational potential
- :math:`a = c_s / \sqrt{4\pi G \rho_c}` is the scale radius

The density contrast relative to the center is:

.. math::
   :label: eq-density-contrast

   \frac{\rho(\xi)}{\rho_c} = e^{-u(\xi)}

**Boundary conditions** at the center (:math:`\xi \to 0`):

.. math::

   u(0) = 0, \quad \frac{du}{d\xi}\bigg|_{\xi=0} = 0


Key Parameters
""""""""""""""

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

The dimensionless mass is computed from the Lane-Emden solution:

.. math::
   :label: eq-be-mass

   m(\xi) = -\xi^2 \frac{du}{d\xi}

which converts to physical mass via :math:`M = m(\xi) \cdot \rho_c \cdot a^3`.


Stability Criterion
"""""""""""""""""""

Bonnor-Ebert spheres have a **critical density contrast**:

.. math::

   \Omega_{\rm crit} \approx 14.04

corresponding to :math:`\xi_{\rm crit} \approx 6.45`.

.. warning::

   Spheres with :math:`\Omega > \Omega_{\rm crit}` are **gravitationally unstable** and will collapse. TRINITY allows such configurations but the user should be aware of the physical implications.

.. code-block:: text

   Bonnor-Ebert Density Profile

   n(r)
    │
    │▓▓▓▓▓▓▓
    │       ▓▓▓
    │          ▓▓▓
    │             ▓▓▓
    │                ▓▓▓
    │                   ▓▓▓▓▓▓▓▓────────────
    │                              nISM
    └────────────────────────────────────────► r
                      rCloud

   Note: Profile is smooth (no flat core like power-law)


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


Variable Summary
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 15 20 15 50
   :header-rows: 1

   * - Symbol
     - Parameter
     - Unit
     - Description
   * - :math:`M_{\rm cloud}`
     - ``mCloud``
     - :math:`M_\odot`
     - Total cloud mass
   * - :math:`n_{\rm core}`
     - ``nCore``
     - cm\ :sup:`-3`
     - Core number density
   * - :math:`r_{\rm core}`
     - ``rCore``
     - pc
     - Core radius (power-law only)
   * - :math:`\alpha`
     - ``densPL_alpha``
     - --
     - Power-law exponent
   * - :math:`\Omega`
     - ``densBE_Omega``
     - --
     - BE density contrast
   * - :math:`n_{\rm ISM}`
     - ``nISM``
     - cm\ :sup:`-3`
     - Ambient ISM density
   * - :math:`r_{\rm cloud}`
     - (computed)
     - pc
     - Cloud outer radius


References
^^^^^^^^^^

.. [Ebert1955] Ebert, R. (1955). "Über die Verdichtung von H I-Gebieten." *Zeitschrift für Astrophysik*, 37, 217. `ADS <https://ui.adsabs.harvard.edu/abs/1955ZA.....37..217E>`_

.. [Bonnor1956] Bonnor, W. B. (1956). "Boyle's Law and gravitational instability." *MNRAS*, 116, 351. `ADS <https://ui.adsabs.harvard.edu/abs/1956MNRAS.116..351B>`_

.. [Rahner2017] Rahner, D., Pellegrini, E. W., Klessen, R. S., & Glover, S. C. O. (2017). "WARPFIELD: A semi-analytic model of HII region expansion." *MNRAS*, 470, 4453. `ADS <https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.4453R>`_

.. [Rahner2018] Rahner, D., Pellegrini, E. W., Glover, S. C. O., & Klessen, R. S. (2018). "WARPFIELD population synthesis." *MNRAS*, 473, 4862. `ADS <https://ui.adsabs.harvard.edu/abs/2018MNRAS.473.4862R>`_


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

TRINITY evolves through distinct phases:

1. **Energy Phase**: Adiabatic bubble expansion with :math:`E_b` evolution
2. **Implicit Energy Phase**: Bubble with cooling/heating balance
3. **Transition Phase**: Handoff from energy-driven to momentum-driven
4. **Momentum Phase**: No bubble, ram pressure from mechanical luminosity only

The transition from energy-driven to momentum-driven expansion is governed by the
shell mass freeze condition and the blending weight evolution.


See Also
--------

- :ref:`sec-parameters` for input parameter specifications
- :ref:`sec-architecture` for code structure and data flow
- :ref:`sec-visualization` for diagnostic plotting tools
- :ref:`sec-trinity-reader` for reading output data
