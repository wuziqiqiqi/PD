.. currentmodule:: clease.montecarlo.mc_evaluator

Monte Carlo Evaluator
=====================

For standard Monte Carlo (MC) runs using the standard :class:`clease.calculator.clease.Clease` cluster expansion (CE) calculator,
this is generally not required. However, it is possible to use the :class:`clease.montecarlo.montecarlo.Montecarlo` class
without the CLEASE calculator, and use a different calculator instead.

In general, if the atoms object has a generic calculator attached, which is not a CLEASE calculator, it will assume it is an ASE calcualtor,
and simply use the ``get_potential_energy`` method of the calculator object. This will also cause a complete re-evaluation of the entire system
whenever a change is proposed in the MC algorithm, which may or may not be desired. The specifics of how to deal with local changes in the
energy evaluation is up to the individual cases, but let's take a look at how to use the ASE EMT calculator with the MC class,
using the :class:`MCEvaluator` class.

An Example
----------

Let's assume we have a system comprised of ``Au``, ``Cu`` and vacancies (in ASE denoted as ``X``).
The EMT calculator is unable to evaluate an atom which is ``X``, however we need to keep track of them anyway
in the Monte Carlo run. We can then create a new MC evaluator, which changes the rules for how we get the energy,
by removing vacancies from the atoms object prior to evaluating the energy.


>>> from clease.montecarlo import MCEvaluator
>>> from ase.calculators.emt import EMT
>>> class MyEvaluator(MCEvaluator):
...     def __init__(self, atoms):
...         super().__init__(atoms)
...         # Have a pre-made calculator instance ready
...         self.calc = EMT()
...     def get_energy(self, applied_changes = None) -> float:
...         # Make a copy of the atoms, and remove all vacancies.
...         atoms_cpy = self.atoms.copy()
...         mask = [atom.index for atom in atoms_cpy if atom.symbol != 'X']
...         atoms_cpy = atoms_cpy[mask]
...         
...         atoms_cpy.calc = self.calc
...         return atoms_cpy.get_potential_energy()  

Note that we overwrite the ``get_energy`` method of the :class:`MCEvaluator`, in order to have custom
rules for the energy evaluation. Let's create an example system to run the MC on:

.. doctest::

    >>> from ase.build import bulk
    >>> atoms = bulk('Au') * (5, 5, 5)
    >>> atoms.symbols[:10] = 'Cu'
    >>> atoms.symbols[10:20] = 'X'
    >>> print(atoms.symbols)
    Cu10X10Au105

We can now run our Monte Carlo:

.. doctest::

    >>> from clease.montecarlo import Montecarlo
    >>> temp = 300  # 300 kelvin
    >>> evaluator = MyEvaluator(atoms)
    >>> mc = Montecarlo(evaluator, temp)
    >>> mc.run(steps=10)

Which successfully now runs our MC simulation on an atoms object using custom energy evaluation rules.
You can write your own custom evaluators to do more complex things, such as utilizing the ``applied_changes``
keyword, to make energy evaluations only consider local changes to the atoms object.

The API
--------

.. autoclass:: MCEvaluator
    :members:
