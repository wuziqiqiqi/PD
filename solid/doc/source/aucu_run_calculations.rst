.. _aucu_run_calculations:
.. module:: clease.tools

Running calculations on generated structures
============================================

For this tutorial, we use :mod:`EMT <ase.calculators.emt>` calculator to
demonstrate how one can run calculations on the structures generated using
CLEAES and update database with the calculation results for further evaluation
of the CE model. Here is a simple example script that runs the calculations
for all structures that are not yet converged

.. doctest::
  :options: +SKIP

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
  ...   atoms.calc = calc
  ...   atoms.get_potential_energy()
  ...   update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)

CLEASE has :func:`update_db` function to update the database entry with the
calculation results. It automatically updates the intial structure entry and
generates a new entry for the final structure. The key-value pairs of the
initial structure entry are updated as:

.. list-table::
   :header-rows: 1

   * - key
     - description
   * - ``converged``
     - True
   * - ``started``
     - *empty*
   * - ``queued``
     - *empty*
   * - ``final_struct_id``
     - ID of the DB entry containing the final converged structure


.. autofunction:: update_db
