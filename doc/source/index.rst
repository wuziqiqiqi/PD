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

Installation
------------

A latest stable version of CLEASE can be installed using the following command

.. code-block:: bash

    pip install clease

.. _conda: https://conda.io

Installation can also be done through `conda`_ via the `conda-forge <https://conda-forge.org/>`_
project:

.. code-block:: bash

   conda install -c conda-forge clease

.. note::
   On Windows, we recommend installing CLEASE with `conda`_,
   in order to simplify the compilation process.

Alternatively, you can install the latest development version of CLEASE by
following the instructions in the `README <https://gitlab.com/computationalmaterials/clease/blob/master/README.md>`_ page.

GUI
---

Most of the standard CE routines can be performed using the graphical user
interface (GUI). The `CLEASE GUI <https://clease-gui.readthedocs.io>`_
is an app based on the jupyter notebook.

.. note::

   The GUI is still under early development, so please don't forget to
   `report any issues <https://gitlab.com/computationalmaterials/clease-gui/-/issues>`_
   if you use the GUI.

Using CLEASE
-------------

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

   releasenotes
   ce_aucu
   metadynamics_sampling
   clease_cli
   import_structures
   api_doc
   benchmarking
   publications
   acknowledgements

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
