.. CLEASE documentation master file, created by
   sphinx-quickstart on Fri Aug 16 15:58:30 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Cluster Expansion in Atomic Simulation Environment (CLEASE)
===========================================================

Cluster expansion (CE) is a widely used method for studying thermondynamic
properties of disordered materials. CLEASE offers:

* Tools to construct a CE model

  * Semi-automatic generation of training set

  * Database to store calculation results

  * Various fitting methods to evaluate the CE model.

* Various flavors of Monte Carlo samplers where one can explore a large
  configurational space in a relatively large simulastion cell.

A latest stable version of CLEASE can be installed using the following command

.. code-block:: bash

    pip install clease

Most of the standard CE routines can be performed using the graphical user
interface (GUI). All of the necessary dependencies for GUI can be installed
by running

.. code-block:: bash

    clease gui-setup

Alternatively, you can install the latest development version of CLEASE by
following the instructions in the `README <https://gitlab.com/computationalmaterials/clease/blob/master/README.md>`_ page.

The method and implementation details of CLEASE are described in the following
publication:

   | J. Chang, D. Kleiven, M. Melander, J. Akola, J. M. Garcia-Lastra and T. Vegge
   | `CLEASE: A versatile and user-friendly implementation of Cluster Expansion method`__
   | Journal of Physics: Condensed Matter

   __ https://doi.org/10.1088/1361-648X/ab1bbc


The use of CLEASE is best learned through tutorials:


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   ce_aucu
   metadynamics_sampling
   ce_aucu_gui_tutorial
   clease_cli
   import_structures
   api_doc
   publications



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


.. The use of CLEASE is best learned through tutorials:

.. .. toctree::
..    :maxdepth: 2

.. A simple tutorial explaining how to set up a database and perform a set of
.. calculations for Cu-Au alloy can be found here: :ref:`ce_aucu_tutorial`