.. _mc_api:

============
Monte Carlo
============

.. contents:: Table of Contents


Canonical MC
=============
The canonical Monte Carlo class has the following API:

.. autoclass:: clease.montecarlo.montecarlo.Montecarlo
    :members:

Semi-grand canonical MC
========================
The semi-grand canonical (SGC) Monte Carlo class:

.. autoclass:: clease.montecarlo.sgc_montecarlo.SGCMonteCarlo
    :members:


Related Objects
===============

All MC classes inherit from the
:class:`~clease.montecarlo.base.BaseMC` interface,
which adds the following methods:

.. autoclass:: clease.montecarlo.base.BaseMC
    :members:

Individual steps from montecarlo iterations return
:class:`~clease.datastructures.mc_step.MCStep` objects:

.. autoclass:: clease.datastructures.mc_step.MCStep
    :members:

Below are some related objects, which may be useful in your Monte Carlo endeavours.

.. toctree::

    ./mc_constraints
    ./mc_observers
    ./mc_evaluator
    ./trial_move_generators
