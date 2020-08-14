.. _aucu_concentration:
.. module:: clease.concentration

Specify the concentration ranges of species
===========================================

The first step in setting up CE in ASE is to specify the types of elements
occupying each basis and their concentration ranges using
:class:`Concentration` class. For AuCu alloys, we consider the entire
composition range of Au\ :sub:`x`\ Cu\ :sub:`1-x` where
:math:`0 \leq x \leq 1`. The :class:`Concentration` object can be created
simply as

>>> from clease.settings import Concentration
>>> conc = Concentration(basis_elements=[['Au', 'Cu']])

because there is no restriction imposed on the concentration range. Note that
a nested list is passed for the ``basis_elements`` argument because
the consituting elements are specified per basis and FCC (crystal structure of
Au\ :sub:`x`\ Cu\ _sub:`1_-_x` for all :math:`0 \leq x \leq 1`) has only one
basis. The initialization automatically creates a linear algebra representation
of the default concentration range constraints. The equality condition of

.. math:: A_\mathrm{eq} = \begin{bmatrix}
                            \begin{bmatrix} 1 & 1 \end{bmatrix}
                          \end{bmatrix}

and

.. math:: b_\mathrm{eq} = \begin{bmatrix} 1 \end{bmatrix}

as well as the lower bound conditions of

.. math:: A_\mathrm{lb} = \begin{bmatrix}
                              \begin{bmatrix}1 & 0\end{bmatrix},
                              \begin{bmatrix}0 & 1\end{bmatrix}
                          \end{bmatrix}

and

.. math:: b_\mathrm{lb} = \begin{bmatrix}0 & 0\end{bmatrix}

are created automatically. The conditions represents the linear
equations

.. math:: A_\mathrm{eq} c_\mathrm{species} = b_\mathrm{eq}

and

.. math:: A_\mathrm{lb} c_\mathrm{species} \geq b_\mathrm{lb},

where the concentration list, :math:`c_\mathrm{species}`, is defined as

.. math:: c_\mathrm{species} =  \begin{bmatrix}
                                  c_\mathrm{Au} & c_\mathrm{Cu}
                                \end{bmatrix}.

The equality condition is then expressed as

.. math:: c_\mathrm{Au} + c_\mathrm{Cu} = 1,

which specifies that elements Au and Cu constitute the entire basis (only one
basis in this case). The lower bound conditions are expressed as

.. math:: c_\mathrm{Au} \geq 0

and

.. math:: c_\mathrm{Cu} \geq 0,

which speicifies that the concentrations of Au and Cu must be greater than or
equal to zero.

The AuCu system presented in this tutorial does not impose any concentration
constraints. However, we demonstrate how one can impose extra constraints by
using an example case where the concentration of interest is
Au\ :sub:`x`\ Cu\ :sub:`1_-_x` where :math:`0 \leq x \leq 0.5`.
The extra concentration constraint can be specified in one of three ways.

The first method is to specify the extra constraint using ``A_eq``, ``b_eq``,
``A_lb`` and ``b_lb``. For this particular case, the extra constraint is
specified using ``A_lb`` and ``b_lb`` arguments as

>>> from clease.settings import Concentration
>>> conc = Concentration(basis_elements=[['Au', 'Cu']], A_lb=[[2, 0]], b_lb=[1])

A list of many examples on how linear systems equations can be used, is found 
:doc:`here </linsys_examples>`.

The second method is to specify the concentration range using formula unit
strings. The :class:`Concentration` class contains
:meth:`~Concentration.set_conc_formula_unit()` method which accepts formula
strings and variable range, which can be invoked as

>>> from clease.settings import Concentration
>>> conc = Concentration(basis_elements=[['Au', 'Cu']])
>>> conc.set_conc_formula_unit(formulas=["Au<x>Cu<1-x>"], variable_range={"x": (0, 0.5)})

The last method is to specify the concentration range each constituting species
using :meth:`~Concentration.set_conc_ranges()` method in :class:`Concentration`
class. The lower and upper bound of species are specified in a nested list in
the same order as the ``basis_elements`` as

>>> from clease.settings import Concentration
>>> conc = Concentration(basis_elements=[['Au', 'Cu']])
>>> conc.set_conc_ranges(ranges=[[(0, 0.5), (0.5, 1)]])

The above three methods yields the same results where :math:`x` is
constrained to :math:`0 \leq x \leq 0.5`.


.. autoclass:: Concentration
  :members: set_conc_formula_unit, set_conc_ranges

