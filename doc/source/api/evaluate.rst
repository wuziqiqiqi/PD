=============
Fitting ECIs
=============

.. contents:: Table of Contents

The Evaluate Class
==================

.. autoclass:: clease.evaluate.Evaluate
    :members:


Fitting ECI's to Non-Energy Properties
======================================

.. note:: It is currently only possible to fit to values stored as key-value pairs in the database,
    i.e. it cannot be the default built-in ``fmax`` or similar properties, yet.
    To get around this, store the desired property as a key-value pair with a (slightly) different name.

.. note:: The desired target property should be stored in the row belonging to the **final** structure.

It is possible to fit ECI's to non-energy properties, and instead use values stored as key-value pairs.
To do this, use the ``prop`` keyword in the evalutate class. As an example, say we already have a database
of completed DFT calculations, and we wanted to fit to the average magnetic moment (why would want to do that
you ask? In this case, for the sake of demonstration!).

Let's assume that this quantity has not already been calculated from our database,
so we first loop through our final structures, find the average magnetic moment, and insert that
quantity back in the database as a key-value pair.

.. code-block:: python

    from ase.db import connect
    import numpy as np

    db = connect("clease.db")  # We assume our database is called 'clease.db'
    # Select all the final structures
    for row in db.select(struct_type="final"):
        atoms = row.toatoms()
        avg_magmom = np.mean(atoms.get_magnetic_moments())
        # Insert the new quantity as a key-value pair.
        db.update(row.id, avg_magmom=avg_magmom)

Now we calculated the average magnetic moment of all our final structures. We can now do a fit on this
new property with our evaluate class, ``Evalutate(..., prop='avg_magmom')`` and then proceeding as normal.
