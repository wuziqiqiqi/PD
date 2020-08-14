.. testsetup::

  from clease.settings import Concentration
  conc = Concentration(basis_elements=[['Au', 'Cu']])

.. _aucu_setting:
.. module:: clease.settings

Specify CE settings
===================

The next step is to specify the settings in which the CE model is constructed.
One of :class:`CEBulk` or :class:`CECrystal` classes is used to specify the
settings. :class:`CEBulk` class is used when the crystal structure is one of
"sc", "fcc", "bcc", "hcp", "diamond", "zincblende", "rocksalt",
"cesiumchloride", "fluorite" or "wurtzite".

Here is how to specify the settings for performing CE on
Au\ :sub:`x`\ Cu\ :sub:`1-x` for all :math:`0 \leq x \leq 1` on FCC lattice
with a lattice constant of 3.8 Å

>>> from clease.settings import CEBulk
>>> settings = CEBulk(crystalstructure='fcc',
...                   a=3.8,
...                   supercell_factor=64,
...                   concentration=conc,
...                   db_name="aucu.db",
...                   max_cluster_size=4,
...                   max_cluster_dia=[6.0, 4.5, 4.5])

.. testcleanup::

  import os
  os.remove("aucu.db")

:class:`CEBulk` internally calls :func:`ase.build.bulk` function to generate a
unit cell. Arguments ``crystalstructure``, ``a``, ``c``, ``covera``, ``u``,
``orthorhombic`` and ``cubic`` are passed to :func:`ase.build.bulk` function to
generate a unit cell from which the supercells are generated. In case where one
prefers to perform CE on a single, fixed size supercell, ``size`` parameter can
be set by passing a list of three integer values (e.g., [3, 3, 3] for a
:math:`3 \times 3 \times 3` supercell). More generally, a ``supercell_factor``
argument is specified to set a threshold on the maximum size of the supercell.

The maximum size of clusters (i.e., number of atoms in a given cluster) and
their maximum diameters are specified using ``max_cluster_size`` and
``max_cluster_dia``, respectively. As empty and one-body clusters do not need
diamters in specifying the clusters, maximum diameters of clusters starting
from two-body clusters are specified in ``max_cluster_dia`` in ascending order.

.. note::
   Several entries are generated in the database file with their names assigned
   as "templates". These templates are used to generate new structures and also
   to calculate their correlation functions.

There are several flavors of cluter expansion formalism in specifying the basis
function for setting the site variable. Three types of basis functions are
currently supported in ASE. The type of basis function can be selected by
passing one of "polynomial", "trigonometric" and "binary_linear" to
``basis_function`` argument. More information on each basis function can be
found in the following articles.

"polynomial":

   | Sanchez, J. M., Ducastelle, F. and Gratias, D. (1984)
   | `Generalized cluster description of multicomponent systems`__
   | Physica A: Statistical Mechanics and Its Applications, 128(1-2), 334-350.

   __ https://doi.org/10.1016/0378-4371(84)90096-7

"trigonometric":

    | van de Walle, A. (2009)
    | `Multicomponent multisublattice alloys, nonconfigurational entropy and other additions to the Alloy Theoretic Automated Toolkit`__
    | Calphad, 33(2), 266-278.

    __ https://doi.org/10.1016/j.calphad.2008.12.005

"binary_linear":

    | Zhang, X. and Sluiter M. (2016)
    | `Cluster expansions for thermodynamics and kinetics of multicomponent alloys.`__
    | Journal of Phase Equilibria and Diffusion 37(1), 44-52.

    __ https://doi.org/10.1007/s11669-015-0427-x


One can alternatively use :class:`CECrystal` class to specify the unit cell of
the system. :class:`CECrystal` takes a more general approach where the unit
cell is specified based on its space group and the positions of unique sites.


.. autoclass:: CEBulk
.. autoclass:: CECrystal

