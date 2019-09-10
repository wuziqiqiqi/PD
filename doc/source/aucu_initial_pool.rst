.. _aucu_tutorial_initial_pool:
.. module:: clease.newStruct


Generating initial structures
=============================


Generating initial pool of structures
-------------------------------------

After the cluster expansion setting is specified, the next step is to generate
initial structures to start training the CE model. New structures for training
CE model are generated using :class:`NewStructures` class, which contains
several methods for generating structures. The initial pool of structures is
generated using :meth:`generate_initial_pool` method as

  >>> from clease import Concentration, CEBulk, NewStructures
  >>> conc = Concentration(basis_elements=[['Au', 'Cu']])
  >>> setting = CEBulk(crystalstructure='fcc',
  ...                  a=3.8,
  ...                  supercell_factor=27,
  ...                  concentration=conc,
  ...                  db_name="aucu_bulk.db",
  ...                  max_cluster_size=4,
  ...                  max_cluster_dia=[6.0, 5.0, 5.0],
  ...                  basis_function='polynomial')
  >>> ns = NewStructures(setting=setting, generation_number=0, struct_per_gen=10)
  >>> ns.generate_initial_pool()

The :meth:`generate_initial_pool` method generates one structure per
concentration where the number of each constituing element is at
maximum/minimum. In the case of AuCu alloy, there are two extrema:
Au and Cu. Consequently, :meth:`generate_initial_pool` generates
two structures for training.

.. note::
   * ``generation_number`` is used to track at which point you generated the
     structures.
   * ``struct_per_gen`` specifies the maximum number of structures to be
     generated for that generation number.

The generated structures are automatically stored in the database with several
key-value pairs specifying their features. The genereated keys are:

.. list-table::
   :header-rows: 1

   * - key
     - description
   * - ``gen``
     - generation number
   * - ``struct_type``
     - ''initial'' for input structures, "final" for converged structures after calculation
   * - ``size``
     - size of the supercell
   * - ``formula_unit``
     - reduced formula unit representation independent of the cell size
   * - ``name``
     - name of the structure (``formula_unit`` followed by a number)
   * - ``converged``
     - Boolean value indicating whether the calculation of the structure is converged
   * - ``queued``
     - Boolean value indicating whether the calculation is queued in the workload manager
   * - ``started``
     - Boolean value indicating whether the calculation has started


Generating random pool of structures
------------------------------------
As we have generated only two structures for training, we can generate more
random structures using :meth:`generate_random_structures` method by altering
the above script with::

    >>> from clease import Concentration, CEBulk, NewStructures
    >>> conc = Concentration(basis_elements=[['Au', 'Cu']])
    >>> setting = CEBulk(crystalstructure='fcc',
    ...                  a=3.8,
    ...                  supercell_factor=27,
    ...                  concentration=conc,
    ...                  db_name="aucu_bulk.db",
    ...                  max_cluster_size=4,
    ...                  max_cluster_dia=[6.0, 5.0, 5.0],
    ...                  basis_function='polynomial')
    >>> ns = NewStructures(setting=setting, generation_number=0,
    ...                    struct_per_gen=10)
    >>> ns.generate_random_structures()

The script generates 8 additional random structures such that there are 10
structures in generation 0. By default, :meth:`generate_random_structures`
method generates a structure with both random size and concentration. If
you prefer to generate random structures with a specific cell size, you
can pass template atoms with desired size. For example, you can force the
new structures to be :math:`3 \times 3 \times 3` supercell by using

    >>> from clease import Concentration, CEBulk, NewStructures
    >>> from ase.db import connect
    >>> conc = Concentration(basis_elements=[['Au', 'Cu']])
    >>> setting = CEBulk(crystalstructure='fcc',
    ...                  a=3.8,
    ...                  supercell_factor=27,
    ...                  concentration=conc,
    ...                  db_name="aucu_bulk.db",
    ...                  max_cluster_size=4,
    ...                  max_cluster_dia=[6.0, 5.0, 5.0],
    ...                  basis_function='polynomial')
    >>> ns = NewStructures(setting=setting, generation_number=0,
    ...                    struct_per_gen=10)

    >>> atoms = connect('aucu_bulk.db').get(id=11).toatoms()
    >>> ns.generate_random_structures(atoms)


.. autoclass:: NewStructures
   :members: generate_initial_pool, generate_random_structures
