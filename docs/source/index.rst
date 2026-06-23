TRINITY
=======

TRINITY is a feedback-driven HII-region evolution code. For a given
giant-molecular-cloud mass, star-formation efficiency, density profile,
and ambient medium, it integrates the time evolution of an expanding
feedback bubble — shell radius, velocity, thermal state, and force
budget — and resolves the phase transitions (energy-driven →
transition → momentum-driven) and stopping fate (stall, dissolution,
escape) of the shell.

TRINITY is distributed under the :doc:`GNU GPL v3 <license>`. If you
use it in published work, please see :doc:`publications` for the
citation and acknowledgement.


Install
-------

Clone the repository and install the Python dependencies::

    git clone https://github.com/JiaWeiTeh/trinity
    cd trinity
    pip install -r requirements.txt

TRINITY is pure Python (no compilation step) and requires Python 3.9
or newer.


First run
---------

From the repository root::

    python run.py param/simple_cluster.param --local

This integrates a small, pre-shipped example (a 1e5 :math:`M_\odot`
cloud at 30% star-formation efficiency, with everything else falling
back to defaults). Outputs land in the directory specified by
``path2output``; the default sentinel ``def_dir`` writes to
``outputs/<model_name>/`` under the current working directory. See
:ref:`sec-running` for the parameter-file syntax, sweep modes, CLI
flags, and output layout.


Contents
--------

.. toctree::
   :maxdepth: 2

   running
   parameters
   trinity_reader
   visualization
   publications
   license
