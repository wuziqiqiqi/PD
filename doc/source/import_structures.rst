Importing Structures
======================

If you have DFT data that are stored in different place/format than the CLEASE database 
(databases, trajectory files, xyz files, etc.), 
CLEASE offers the possibility of importing those structures. The only thing that needs to be provided
is the **initial** (e.g. non-relaxed structure where all atoms are on ideal sites) and the total energy
associated with it. Note that the total energy can be one of the relaxed structure. To show how this
feature can be used we generate an example dataset using ASE's EMT calculator and store them in a trajectory
file.

.. doctest::

    :options: +SKIP
    >>> from ase.calculators.emt import EMT
    >>> from ase.build import bulk
    >>> from ase.io.trajectory import TrajectoryWriter
    >>> writer_initial = TrajectoryWriter("initial.traj")
    >>> writer_final = TrajectoryWriter("final.traj")
    >>> for i in range(10):
    ...     atoms = bulk("Au", a=4.05)*(3, 3, 3)
    ...     writer_initial.write(atoms)
    ...     calc = EMT()
    ...     atoms.calc = calc
    ...     en = atoms.get_potential_energy()
    ...     writer_final.write(atoms)


Next, we want to import these data into CLEASE. First, we create the settings

.. doctest::

    :options: +SKIP
    >>> from clease.structgen import NewStructures
    >>> from clease.settings import CEBulk, Concentration
    >>> settings = CEBulk(
    ...     Concentration(basis_elements=[['Au', 'Cu']]), 
    ...     crystalstructure='fcc', a=4.05, db_name="imported.db")
    >>> new_struct = NewStructures(settings)

    Next, we load our structures

    >>> from ase.io.trajectory import TrajectoryReader
    >>> reader_init = TrajectoryReader("initial.traj")
    >>> reader_final = TrajectoryReader("final.traj")
    >>> for i in range(len(reader_init)):
    ...     initial = reader_init[i]
    ...     final = reader_final[i]
    ...     new_struct.insert_structure(init_struct=initial, final_struct=final)

Note that it is important that the final structure has energy. In case you have stored the structures
in a way that the energy is not added to the structures when it is loaded, add the energy to the final
structure via a **SinglePointCalculator**. Furthermore, if you only have the initial structure (and not the final),
you can perfectly fine just replace the final structure with a copy of the initial.

.. testcleanup::

    import os
    os.remove("initial.traj")
    os.remove("final.traj")
    os.remove("imported.db")
