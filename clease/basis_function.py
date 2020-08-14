"""Module for setting up pseudospins and basis functions."""
import numpy as np
import math
from clease.gramSchmidthMonomials import GramSchmidtMonimial
from typing import List, Dict, Optional

__all__ = ('BasisFunction', 'Polynomial', 'Trigonometric', 'BinaryLinear')


class BasisFunction(object):
    """Base class for all Basis Functions."""

    def __init__(self, unique_elements: List[str]) -> None:
        self.name = "generic"
        self._unique_elements = sorted(unique_elements)
        if self.num_unique_elements < 2:
            raise ValueError("Systems must have more than 1 type of element.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BasisFunction):
            return False
        return self.name == other.name and \
            self.unique_elements == other.unique_elements

    @property
    def unique_elements(self) -> List[str]:
        return self._unique_elements

    @unique_elements.setter
    def unique_elements(self, elements):
        self._unique_elements = sorted(elements)

    @property
    def num_unique_elements(self) -> int:
        return len(self.unique_elements)

    @property
    def spin_dict(self) -> Dict[str, int]:
        return self.get_spin_dict()

    @property
    def basis_functions(self) -> List[Dict[str, float]]:
        return self.get_basis_functions()

    def get_spin_dict(self):
        """Get spin dictionary."""
        raise NotImplementedError("get_spin_dict has to be implemented in derived classes!")

    def get_basis_functions(self):
        """Get basis function."""
        raise NotImplementedError(("get_basis_functions has to be implemented "
                                   "in derived classes!"))

    def customize_full_cluster_name(self, full_cluster_name: str) -> str:
        """Customize the full cluster names. Default is to do nothing."""
        return full_cluster_name

    def todict(self) -> dict:
        """
        Create a dictionary representation of the basis function class
        """
        return {'name': self.name, 'unique_elements': self.unique_elements}


class Polynomial(BasisFunction):
    """Pseudospin and basis function from Sanchez et al.

    Sanchez, J. M., Ducastelle, F. and Gratias, D. (1984).
    Generalized cluster description of multicomponent systems.
    Physica A: Statistical Mechanics and Its Applications, 128(1-2), 334-350.
    """

    def __init__(self, unique_elements: List[str]):
        BasisFunction.__init__(self, unique_elements)
        self.name = "polynomial"

    def get_spin_dict(self) -> Dict[str, int]:
        """Define pseudospins for all consistuting elements."""
        gram_schmidt = GramSchmidtMonimial(self.num_unique_elements)
        spin_values = gram_schmidt.values
        spin_dict = {}
        for x in range(self.num_unique_elements):
            spin_dict[self.unique_elements[x]] = spin_values[x]
        return spin_dict

    def get_basis_functions(self) -> List[Dict[str, float]]:
        """Create basis functions to guarantee the orthonormality."""
        gram_schmidt = GramSchmidtMonimial(self.num_unique_elements)
        gram_schmidt.build()
        return gram_schmidt.basis_functions(self.unique_elements)


class Trigonometric(BasisFunction):
    """Pseudospin and basis function from van de Walle.

    van de Walle, A. (2009).
    Multicomponent multisublattice alloys, nonconfigurational entropy and other
    additions to the Alloy Theoretic Automated Toolkit. Calphad, 33(2),
    266-278.
    """

    def __init__(self, unique_elements: List[str]):
        BasisFunction.__init__(self, unique_elements)
        self.name = "trigonometric"

    def get_spin_dict(self) -> Dict[str, int]:
        """Define pseudospins for all consistuting elements."""
        spin_values = list(range(self.num_unique_elements))
        spin_dict = {}
        for x in range(self.num_unique_elements):
            spin_dict[self.unique_elements[x]] = spin_values[x]
        return spin_dict

    def get_basis_functions(self) -> List[Dict[str, float]]:
        """Create basis functions to guarantee the orthonormality."""
        alpha = list(range(1, self.num_unique_elements))
        bf_list = []

        for a in alpha:
            bf = {}
            for key, value in self.spin_dict.items():
                var = 2 * np.pi * math.ceil(a / 2.) * value
                var /= self.num_unique_elements
                if a % 2 == 1:
                    bf[key] = -np.cos(var) + 0.
                else:
                    bf[key] = -np.sin(var) + 0.

            # normalize the basis function
            sum = 0
            for key, value in self.spin_dict.items():
                sum += bf[key] * bf[key]
            normalization_factor = np.sqrt(self.num_unique_elements / sum)

            for key, value in bf.items():
                bf[key] = value * normalization_factor

            bf_list.append(bf)

        return bf_list


def _kronecker(i: int, j: int) -> int:
    """Kronecker delta function."""
    if i == j:
        return 1
    return 0


class BinaryLinear(BasisFunction):
    """Pseudospin and basis function from Zhang and Sluiter.

    Zhang, X. and Sluiter M.
    Cluster expansions for thermodynamics and kinetics of multicomponent
    alloys.
    Journal of Phase Equilibria and Diffusion 37(1) 44-52.
    """

    def __init__(self, unique_elements: List[str], redundant_element: Optional[str] = "auto"):
        if redundant_element == "auto":
            self.redundant_element = sorted(unique_elements)[0]
        else:
            self.redundant_element = redundant_element
        BasisFunction.__init__(self, unique_elements)
        self.name = "binary_linear"

    def get_spin_dict(self) -> Dict[str, int]:
        """Define pseudospins for all consistuting elements."""
        spin_values = list(range(self.num_unique_elements))
        spin_dict = {}
        for x in range(self.num_unique_elements):
            spin_dict[self.unique_elements[x]] = spin_values[x]
        return spin_dict

    def get_basis_functions(self) -> List[Dict[str, float]]:
        """Create orthonormal basis functions.

        Due to the constraint that any site is occupied by exactly one element,
        we only need to track N-1 species if there are N species.
        Hence, the first element specified is redundant, and will not
        have a basis function.
        """
        bf_list = []
        num_bf = self.num_unique_elements
        for bf_num in range(num_bf):
            if self.unique_elements[bf_num] == self.redundant_element:
                continue
            new_bf = {
                symb: float(_kronecker(i, bf_num)) for i, symb in enumerate(self.unique_elements)
            }
            bf_list.append(new_bf)
        return bf_list

    def _decoration2element(self, dec_num: int) -> str:
        """Get the element with its basis function equal to 1."""
        bf = self.basis_functions[dec_num]
        for k, v in bf.items():
            if v == 1:
                return k
        raise ValueError("Did not find any element where the value is 1.")

    def customize_full_cluster_name(self, full_cluster_name: str) -> str:
        """Translate the decoration number to element names."""
        dec = full_cluster_name.rsplit("_", 1)[1]
        name = full_cluster_name.rsplit("_", 1)[0]
        new_dec = ""
        for decnum in dec:
            element = self._decoration2element(int(decnum))
            new_dec += f"{element}"
        return name + "_" + new_dec

    def todict(self) -> dict:
        """
        Creates a dictionary representation of the class
        """
        dct_rep = BasisFunction.todict(self)
        dct_rep['redundant_element'] = self.redundant_element
        return dct_rep
