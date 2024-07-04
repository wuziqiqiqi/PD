.. _basis_function_api:

Basis Functions
===============

Each cluster is defined on a set of cluster functions, which is expanded on a set of single-site basis functions.
The basis function obeys the orthogonality condition

.. math:: \frac{1}{M} \sum _{s_i=-m}^m \Theta _n (s_i) \Theta_{n'}(s_i) = \delta _{nn'}

For more information, please see the `CLEASE paper <https://doi.org/10.1088/1361-648X/ab1bbc>`_.
CLEASE implements three different basis functions:
:class:`~clease.basis_function.Polynomial`,
:class:`~clease.basis_function.Trigonometric` and
:class:`~clease.basis_function.BinaryLinear`.



.. autoclass:: clease.basis_function.Polynomial
    :members:

.. autoclass:: clease.basis_function.Trigonometric
    :members:

.. autoclass:: clease.basis_function.BinaryLinear
    :members:

All three basis functions inherit from the same base abstract base interface:

.. autoclass:: clease.basis_function.BasisFunction
    :members:
