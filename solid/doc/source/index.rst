.. CLEASE documentation master file, created by
   sphinx-quickstart on Fri Aug 16 15:58:30 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Cluster Expansion in Atomic Simulation Environment (CLEASE)
===========================================================

Cluster expansion (CE) is a widely used method for studying thermondynamic
properties of disordered materials. CLEASE is a cluster expansion code which strives
to be highly flexible and customizable, which also offering a wide range of useful tools,
such as:

* Tools to construct a CE model

  * Semi-automatic :ref:`structure generation <structgen_api>` for constructing training data,
    such as random, ground-state and probe structures.

  * Database for storing calculation results.

  * Multiple basis functions for the CE model to choose from:
    :class:`~clease.basis_function.Polynomial`,
    :class:`~clease.basis_function.Trigonometric` or
    :class:`~clease.basis_function.BinaryLinear`.

  * Many methods for parameterization :ref:`fitting <fitting_api>` and
    evaluating the CE model, such as :class:`~clease.regression.regression.Lasso`
    :class:`~clease.regression.regression.Tikhonov`, :class:`~clease.regression.physical_ridge.PhysicalRidge`
    and :class:`~clease.regression.ga_fit.GAFit`.

  * Tools for :ref:`easily visualizing <plot_post_process>` the accuracy
    of your CE model, and interact with the plots e.g. when made in a Jupyter
    notebook.

* Various flavors of Monte Carlo samplers where one can explore a large
  configurational space in a large simulation cell

  * Canonical and semi-grand canonical :ref:`Monte Carlo schemes <mc_api>`.

  * Flexible customization options for restricting the model during MC runs.
    CLEASE provides a :ref:`number of constraints <mc_constraints>`,
    but it is also easy to :ref:`implement custom constraints <implementing your own constraints>`.

  * Use one our :ref:`pre-made observers <mc_observers>` to collect thermodynamic
    data about your system during an MC run, or
    :ref:`write your own <implementing your own observer>`.


and much more. A tutorial of how to use CLEASE can be found in our :ref:`AuCu example <ce_aucu_tutorial>`.


GUI
---
.. _clease-gui:

Most of the standard CE routines can be performed using the graphical user
interface (GUI). The `CLEASE GUI <https://clease-gui.readthedocs.io>`_
is an app based on the jupyter notebook.
Please remember to `report any issues <https://gitlab.com/computationalmaterials/clease-gui/-/issues>`_
to the developers.


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

Using CLEASE
-------------

The method and implementation details of CLEASE are described in the following
publication:

   | J. Chang, D. Kleiven, M. Melander, J. Akola, J. M. Garcia-Lastra and T. Vegge
   | `CLEASE: A versatile and user-friendly implementation of Cluster Expansion method`__
   | Journal of Physics: Condensed Matter

   __ https://doi.org/10.1088/1361-648X/ab1bbc


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   releasenotes
   ce_aucu
   metadynamics_sampling
   clease_cli
   import_structures
   api_doc
   parallelization
   benchmarking
   publications
   acknowledgements
