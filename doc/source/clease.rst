.. module:: ase.clease
   :synopsis: Cluster Expansion in ASE

=================
Cluster Expansion
=================

Cluster expansion (CE) has been widely used for studying thermondynamic
properties of disordered materials such as alloys and oxides.
CE is typically coupled with a Monte Carlo sampler where one can explore a
large configurational space in a relatively large simulation cell once the
CE model is trained.

The CE in ASE does not require any additional external packages.
However, installing `Numba`_ package is strongly encouraged for faster speed,
which is desirable when working with complex systems (e.g., beyond simple
binary alloys).

.. _Numba: http://numba.pydata.org/

The method and implementation details are described in the following
publication:

   | J. Chang, D. Kleiven, M. Melander, J. Akola, J. M. Garcia-Lastra and T. Vegge
   | `CLEASE: A versatile and user-friendly implementation of Cluster Expansion method`__
   | Journal of Physics: Condensed Matter

   __ https://doi.org/10.1088/1361-648X/ab1bbc


The use of CE in ASE is best learned through tutorials:

.. toctree::
   :maxdepth: 1

A simple tutorial explaining how to set up a database and perform a set of
calculations for Cu-Au alloy can be found here: :ref:`ce_aucu_tutorial`
