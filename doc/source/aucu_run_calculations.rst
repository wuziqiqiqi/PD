.. _aucu_tutorial_run_calculations:
.. module:: clease.tools

Running calculations on generated structures
============================================

For this tutorial, we use :mod:`EMT <ase.calculators.emt>` calculator to
demonstrate how one can run calculations on the structures generated using
CLEAES and update database with the calculation results for further evaluation
of the CE model. Here is a simple example script that runs the calculations
for all structures that are not yet converged

  >>> from ase.calculators.emt import EMT
  >>> from ase.db import connect
  >>> from clease.tools import update_db
  >>> calc = EMT()
  >>> db_name = "aucu.db"
  >>> db = connect(db_name)
  >>>
  >>> # Run calculations for all structures that are not converged.
  >>> for row in db.select(converged=False):
  ...   atoms = row.toatoms()
  ...   atoms.set_calculator(calc)
  ...   atoms.get_potential_energy()
  ...   update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)

CLEASE has :func:`update_db` function to update the database entry with the
calculation results. It automatically updates the intial structure entry and
generates a new entry for the final structure. The key-value pairs of the
initial structure entry are updated as::

.. list-table::
   :header-rows: 1

   * - key
     - value
   * - ``converged``
     - True
   * - ``started``
     - *empty*
   * - ``queued``
     - *empty*
   * - ``final_struct_id``
     - ID of the DB entry containing the final converged structure


.. def update_db(uid_initial=None, final_struct=None, db_name=None,
..               custom_kvp_init={}, custom_kvp_final={}):


After the cluster expansion setting is specified, the next step is to generate
initial structures to start training the CE model. New structures for training
CE model are generated using :class:`NewStructures` class, which contains
several methods for generating structures. The initial pool of structures is
generated using :meth:`generate_initial_pool` method as::

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
  >>> ns.generate_initial_pool()

The :meth:`generate_initial_pool` method generates one structure per
concentration where the number of each constituing element is at
maximum/minimum. In the case of AuCu alloy, there are two extrema:
Au and Cu. Consequently, :meth:`generate_initial_pool` generates
two structures for training.

.. note::
   * ``generation_number`` is used to track at which point you generated the
     structures, and each generated structure has a corresponding ``gen``
     key-value pair in the DB entry.
   * ``struct_per_gen`` specifies the maximum number of structures to be
     generated for that generation number.


Generating random pool of structures
------------------------------------
As we have generated only two structures for training, we can generate more
random structures using :meth:`generate_random_structures` method by altering
the above script with

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
new structures to be :math:`3 \times 3 \times 3` supercell by using::

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
    >>>
    >>> atoms = connect('aucu_bulk.db').get(id=11).toatoms()
    >>> ns.generate_random_structures(atoms)



.. autofunction:: update_db
