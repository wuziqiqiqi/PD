.. _stoi-constraints:

Stoichiometric Constraints
---------------------------

The most flexible method of imposing stoichiometric constraints in CLEASE 
is to use linear systems of equations. Here, you can find a list of examples
of how different constraints can be imposed. In CLEASE a linear system of 
equations with the structure shown below

.. figure:: resources/linear_system.svg
    :width: 50%
    :align: center

The number of sublattice concentration is simply the length of the *flattened*
version of *basis_elements* that is passed to the *Concentration* class. Therefore,
if :code:`basis_element = [['Au', 'Cu'], ['Cu', 'X]]` there will be to Cu concentrations
you can restrict; one for each sublattice. The total number of sublattice concentrations
in the example above is 4. Hence, all rows of the matrix has 4 columns. CLEASE has two
types of constraints: **equality** and **lower bound**. Equality constraints are passed
via :code:`A_eq` and :code:`b_eq` arguments in the *Concentration* class, and lower bound 
constraints are passed via :code:`A_lb` and :code:`b_lb`. For lower bound constraints, the 
equality sign in the figure is replaced by a *larger or equal than*-symbol. Note that upper 
bound constraints can trivially be converted to a lower bound constraint by multiplying the 
equation by -1. Finally, the example below shows how you can generate random concentrations 
**satisfying** your constraints. The list passed to the function is the number of sites in 
each sublattice.

>>> import numpy as np
>>> np.random.seed(0)  # Set a seed for consistent tests
>>> from clease.settings import Concentration

Binary System With One Basis
=============================

>>> basis_elements = [['Au', 'Cu']]

This is a system where we have the :code:`basis_elements=[['Au', 'Cu']]`.

1. Force the Au concentration to be equal to the Cu concentration

    >>> A_eq = [[1.0, -1.0]]
    >>> b_eq = [0.0]
    >>> conc = Concentration(basis_elements=basis_elements, A_eq=A_eq, b_eq=b_eq)
    >>> for i in range(10):
    ...     x = conc.get_random_concentration([20])
    ...     assert np.abs(x[0] - x[1]) < 1e-10

2. Force number of Au atoms to be larger than 12

    >>> A_lb = [[20, 0.0]]
    >>> b_lb = [12]
    >>> conc = Concentration(basis_elements=basis_elements, A_lb=A_lb, b_lb=b_lb)
    >>> for i in range(10):
    ...    x = conc.get_random_concentration([20])
    ...    assert round(20*x[0]) >= 12

Two sublattices
================

>>> basis_elements = [['Li', 'V'], ['O', 'F']]

1. Force the concentration of O to be twice the concentration of F

    >>> A_eq = [[0.0, 0.0, -1.0, 2.0]]
    >>> b_eq = [0.0]
    >>> conc = Concentration(basis_elements=basis_elements, A_eq=A_eq, b_eq=b_eq)
    >>> for i in range(10):
    ...     x = conc.get_random_concentration([18, 18])
    ...     assert abs(x[2] - 2*x[3]) < 1e-10

2. Li concentration larger than 0.2 and O concentration smaller than 0.7

    >>> A_lb = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0]]
    >>> b_lb = [0.2, -0.7]
    >>> conc = Concentration(basis_elements=basis_elements, A_lb=A_lb, b_lb=b_lb)
    >>> for i in range(10):
    ...    x = conc.get_random_concentration([18, 18])
    ...    assert x[0] >= 0.2 and x[2] < 0.7

