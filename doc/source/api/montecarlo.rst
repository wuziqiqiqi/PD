============
Monte Carlo
============

.. contents:: Table of Contents


The Monte Carlo class
=====================

The canonical Monte Carlo class has the following API:

.. autoclass:: clease.montecarlo.montecarlo.Montecarlo
    :members:


Additionally, the montecarlo class inherits from the
:class:`~clease.montecarlo.base.BaseMC` class,
which adds the following methods:

.. autoclass:: clease.montecarlo.base.BaseMC
    :members:

Individual steps from montecarlo iterations return
:class:`~clease.datastructures.mc_step.MCStep` objects:

.. autoclass:: clease.datastructures.mc_step.MCStep
    :members:


Related Objects
---------------

Below are some related objects, which may be useful in your Monte Carlo endeavours.

.. toctree::

    ./mc_constraints
    ./mc_observers
    ./mc_evaluator
    ./trial_move_generators
