"""Definition of ClusterExpansionSettings Class.

This module defines the base-class for storing the settings for performing
Cluster Expansion in different conditions.
"""
from copy import deepcopy
from typing import List, Dict, Optional, Union

import numpy as np
from ase.db import connect
from ase import Atoms

from clease.tools import wrap_and_sort_by_position
from clease.basis_function import BasisFunction
from clease.template_atoms import TemplateAtoms
from clease import AtomsManager
from clease.cluster import ClusterList, ClusterManager
from clease.template_filters import ValidConcentrationFilter
from clease.basis_function import Polynomial, Trigonometric, BinaryLinear
from . import Concentration

__all__ = ('ClusterExpansionSettings',)


class ClusterExpansionSettings:
    """Base class for all Cluster Expansion settings."""

    def __init__(self,
                 prim: Atoms,
                 concentration: Concentration,
                 size: Optional[int] = None,
                 supercell_factor: Optional[int] = 27,
                 db_name: str = 'clease.db',
                 max_cluster_size: int = 4,
                 max_cluster_dia: Union[float, List[float]] = 5.0) -> None:
        self.kwargs = {
            'size': size,
            'supercell_factor': supercell_factor,
            'db_name': db_name,
            'max_cluster_size': max_cluster_size,
            'max_cluster_dia': deepcopy(max_cluster_dia)
        }
        self._include_background_atoms = False
        self.trans_matrix = None
        self.concentration = self._get_concentration(concentration)
        self.cluster_list = ClusterList()
        self.basis_elements = deepcopy(self.concentration.basis_elements)
        self._check_first_elements()
        self.db_name = db_name
        self.size = to_3x3_matrix(size)

        self.prim_cell = prim
        self._tag_prim_cell()
        self._store_prim_cell()

        self.cluster_mng = ClusterManager(self.prim_no_bkg())

        prim_mng = AtomsManager(prim)
        prim_ind_by_basis = prim_mng.index_by_symbol([x[0] for x in self.basis_elements])
        conc_filter = ValidConcentrationFilter(concentration, prim_ind_by_basis)

        self.template_atoms = TemplateAtoms(self.prim_cell,
                                            supercell_factor=supercell_factor,
                                            size=self.size,
                                            skew_threshold=40,
                                            filters=[conc_filter])

        self.atoms_mng = AtomsManager(None)

        self.max_cluster_size = max_cluster_size
        self.max_cluster_dia = self._format_max_cluster_dia(max_cluster_dia)
        self.cluster_mng.build(max_size=self.max_cluster_size, max_cluster_dia=self.max_cluster_dia)

        self.set_active_template(atoms=self.template_atoms.weighted_random_template())

        unique_element_no_bkg = self.unique_element_without_background()
        self._basis_func_type = Polynomial(unique_element_no_bkg)

        if len(self.basis_elements) != self.num_basis:
            raise ValueError("list of elements is needed for each basis")

    @property
    def atoms(self) -> Atoms:
        return self.atoms_mng.atoms

    @property
    def all_elements(self) -> List[str]:
        return sorted([item for row in self.basis_elements for item in row])

    @property
    def num_elements(self) -> int:
        return len(self.all_elements)

    @property
    def unique_elements(self) -> List[str]:
        return sorted(list(set(deepcopy(self.all_elements))))

    @property
    def num_unique_elements(self) -> int:
        return len(self.unique_elements)

    @property
    def ref_index_trans_symm(self) -> List[int]:
        return [i[0] for i in self.index_by_sublattice]

    @property
    def skew_threshold(self):
        return self.template_atoms.skew_threshold

    @skew_threshold.setter
    def skew_threshold(self, threshold: int) -> None:
        '''
        Maximum acceptable skew level (ratio of max and min diagonal of the
        Niggli reduced cell)
        '''
        self.template_atoms.skew_threshold = threshold

    @property
    def background_indices(self) -> List[int]:
        """Get indices of the background atoms."""
        # check if any basis consists of only one element type
        basis = [i for i, b in enumerate(self.basis_elements) if len(b) == 1]

        bkg_indices = []
        for b_indx in basis:
            bkg_indices += self.index_by_basis[b_indx]
        return bkg_indices

    @property
    def include_background_atoms(self) -> bool:
        return self._include_background_atoms

    @include_background_atoms.setter
    def include_background_atoms(self, value: bool) -> None:
        if value == self._include_background_atoms:
            return
        self._include_background_atoms = value
        if self._include_background_atoms:
            self.cluster_mng = ClusterManager(self.prim_cell)
        else:
            self.cluster_mng = ClusterManager(self.prim_no_bkg())

        self.cluster_mng.build(max_size=self.max_cluster_size, max_cluster_dia=self.max_cluster_dia)
        self.create_cluster_list_and_trans_matrix()
        self.basis_func_type.unique_elements = \
            self.unique_element_without_background()

    @property
    def spin_dict(self) -> Dict[str, float]:
        return self.basis_func_type.spin_dict

    @property
    def basis_functions(self):
        return self.basis_func_type.basis_functions

    @property
    def ignore_background_atoms(self) -> bool:
        return not self.include_background_atoms

    @property
    def multiplicity_factor(self) -> Dict[str, float]:
        """Return the multiplicity factor of each cluster."""
        num_sites_in_group = [len(x) for x in self.index_by_sublattice]
        return self.cluster_list.multiplicity_factors(num_sites_in_group)

    @property
    def all_cf_names(self) -> List[str]:
        num_bf = len(self.basis_functions)
        return self.cluster_list.get_all_cf_names(num_bf)

    @property
    def num_cf(self) -> int:
        """Return the number of correlation functions."""
        return len(self.all_cf_names)

    @property
    def index_by_basis(self) -> List[List[int]]:
        first_symb_in_basis = [x[0] for x in self.basis_elements]
        return self.atoms_mng.index_by_symbol(first_symb_in_basis)

    @property
    def index_by_sublattice(self) -> List[List[int]]:
        return self.atoms_mng.index_by_tag()

    @property
    def num_basis(self) -> int:
        return len(self.basis_elements)

    @property
    def basis_func_type(self):
        return self._basis_func_type

    @property
    def num_active_sublattices(self) -> int:
        """Number of active sublattices"""
        unique_no_bkg = self.unique_element_without_background()
        active_sublattices = 0
        for basis in self.concentration.orig_basis_elements:
            if basis[0] in unique_no_bkg:
                active_sublattices += 1
        return active_sublattices

    @property
    def ignored_species_and_conc(self) -> Dict[str, float]:
        """
        Return the ignored species and their concentrations normalised to the total number
        of atoms.
        """
        unique_no_bkg = self.unique_element_without_background()
        orig_basis = self.concentration.orig_basis_elements
        nsub_lattices = len(orig_basis)  # Number of sub-lattices
        ignored = {}
        for basis in orig_basis:
            elem = basis[0]
            if elem not in unique_no_bkg:
                if len(basis) != 1:
                    raise ValueError(("Ignored sublattice contains multiple elements -"
                                      "this does not make any sense"))
                if elem not in ignored:
                    ignored[elem] = 1.0 / nsub_lattices
                else:
                    # This element is already on one of the ignored background here we
                    # accumulate the concnetration
                    ignored[elem] += 1.0 / nsub_lattices
        return ignored

    @property
    def atomic_concentration_ratio(self) -> float:
        """
        Ratio between true concentration (normalised to atoms) and the internal concentration used.
        For example, if one of the two basis is fully occupied, and hence ignored internally, the
        internal concentration is half of the actual atomic concentration.
        """
        return self.num_active_sublattices / len(self.concentration.orig_basis_elements)

    @basis_func_type.setter
    def basis_func_type(self, bf_type):
        """
        Type of basis function to use.
        It should be one of "polynomial", "trigonometric" or "binary_linear"
        """
        unique_element = self.unique_element_without_background()

        if isinstance(bf_type, BasisFunction):
            if bf_type.unique_elements != sorted(unique_element):
                raise ValueError("Unique elements in BasisFunction instance "
                                 "is different from the one in settings")
            self._basis_func_type = bf_type
        elif isinstance(bf_type, str):
            if bf_type.lower() == 'polynomial':
                self._basis_func_type = Polynomial(unique_element)
            elif bf_type.lower() == 'trigonometric':
                self._basis_func_type = Trigonometric(unique_element)
            elif bf_type.lower() == "binary_linear":
                self._basis_func_type = BinaryLinear(unique_element)
            else:
                msg = f"basis function type {bf_type} is not supported."
                raise ValueError(msg)
        else:
            raise ValueError("basis_function has to be an instance of BasisFunction or a string")

    def to_dict(self) -> Dict:
        return {
            'kwargs': self.kwargs,
            'include_background_atoms': self.include_background_atoms,
            'skew_threshold': self.skew_threshold,
            'basis_func_type': self.basis_func_type.todict()
        }

    def _get_concentration(self, concentration):
        if isinstance(concentration, Concentration):
            conc = concentration
        elif isinstance(concentration, dict):
            conc = Concentration.from_dict(concentration)
        else:
            raise TypeError("concentration has to be either dict or instance of Concentration")
        self.kwargs["concentration"] = conc.to_dict()
        return conc

    def prim_no_bkg(self):
        """
        Return an instance of the primitive cell where the background indices
        has been removed
        """
        prim = self.prim_cell.copy()
        bg_syms = self.get_bg_syms()
        delete = []
        for atom in prim:
            if atom.symbol in bg_syms:
                delete.append(atom.index)

        delete.sort(reverse=True)
        for i in delete:
            del prim[i]
        return prim

    def get_bg_syms(self):
        """
        Return the symbols in the basis where there is only one element
        """
        return set(x[0] for x in self.basis_elements if len(x) == 1)

    def unique_element_without_background(self):
        """Remove background elements."""
        if self.include_background_atoms:
            bg_sym = set()
        else:
            bg_sym = self.get_bg_syms()

            # Remove bg_syms that are also present in basis with more than one
            # element
            for elems in self.basis_elements:
                if len(elems) == 1:
                    continue
                to_be_removed = set()
                for s in bg_sym:
                    if s in elems:
                        to_be_removed.add(s)

                bg_sym -= to_be_removed

        unique_elem = set()
        for x in self.basis_elements:
            unique_elem.update(x)
        return list(unique_elem - bg_sym)

    def prepare_new_active_template(self, template):
        """Prepare necessary data structures when setting new template."""
        self.size = template.info['size']
        self.atoms_mng.atoms = template

    def set_active_template(self, atoms=None):
        """Set a new template atoms object."""
        if atoms is not None:
            template = self.template_atoms.get_template_matching_atoms(atoms=atoms)
        else:
            template = self.template_atoms.weighted_random_template()

        template = wrap_and_sort_by_position(template)

        if atoms is not None:
            # Check that the positions of the generated template
            # matches the ones in the passed object
            atoms = wrap_and_sort_by_position(atoms)
            if not np.allclose(template.get_positions(), atoms.get_positions()):
                raise ValueError(f"Inconsistent positions. Passed object\n"
                                 f"{atoms.get_positions()}\nGenerated template"
                                 f"\n{template.get_positions()}")
        self.prepare_new_active_template(template)
        self.create_cluster_list_and_trans_matrix()

    def _tag_prim_cell(self):
        """
        Add a tag to all the atoms in the unit cell to track the sublattice.
        Tags are added such that that the lowest tags corresponds to "active"
        sites, while the highest tags corresponds to background sites. An
        example is a system having three sublattices that can be occupied by
        more than one species, and two sublattices that can be occupied by
        only one species. The tags 0, 1, 2 will then be assigned to the three
        sublattices that can be occupied by more than one species, and the two
        remaining lattices will get the tag 3, 4.
        """
        bg_sym = self.get_bg_syms()
        tag = 0

        # Tag non-background elements first
        for atom in self.prim_cell:
            if atom.symbol not in bg_sym:
                atom.tag = tag
                tag += 1

        # Tag background elements
        for atom in self.prim_cell:
            if atom.symbol in bg_sym:
                atom.tag = tag
                tag += 1

    def _store_prim_cell(self):
        """Store unit cell to the database. Returns the id of primitive cell in the database"""
        with connect(self.db_name) as db:
            shape = self.prim_cell.get_cell_lengths_and_angles()
            for row in db.select(name='primitive_cell'):
                uc_shape = row.toatoms().get_cell_lengths_and_angles()
                if np.allclose(shape, uc_shape):
                    return row.id

            uid = db.write(self.prim_cell, name='primitive_cell')
        return uid  # Ensure connection is closed before returning

    def _get_prim_cell(self):
        raise NotImplementedError("This function has to be implemented in in derived classes.")

    def _format_max_cluster_dia(self, max_cluster_dia):
        """Get max_cluster_dia in numpy array form"""
        # max_cluster_dia is list or array
        if isinstance(max_cluster_dia, (list, np.ndarray, tuple)):
            # Length should be either max_cluster_size+1 or max_cluster_size-1
            mcd = np.array(max_cluster_dia, dtype=float)
            if len(max_cluster_dia) == self.max_cluster_size + 1:
                for i in range(2):
                    mcd[i] = 0.
            elif len(max_cluster_dia) == self.max_cluster_size - 1:
                mcd = np.insert(mcd, 0, [0., 0.])
            else:
                raise ValueError("Invalid length for max_cluster_dia.")
        # max_cluster_dia is int or float
        elif isinstance(max_cluster_dia, (int, float)):
            mcd = np.ones(self.max_cluster_size - 1, dtype=float) * max_cluster_dia
            mcd = np.insert(mcd, 0, [0., 0.])
        # Case for *None* or something else
        else:
            raise TypeError("max_cluster_dia is of wrong type, got: {}".format(
                type(max_cluster_dia)))
        return mcd.round(decimals=3)

    def _get_atoms(self):
        """Create atoms with a user-specified size."""
        atoms = self.prim_cell.copy() * self.size
        return wrap_and_sort_by_position(atoms)

    def create_cluster_list_and_trans_matrix(self):
        at_cpy = self.atoms
        self.cluster_list = self.cluster_mng.info_for_template(at_cpy)
        self.trans_matrix = self.cluster_mng.translation_matrix(at_cpy)

    def view_clusters(self):
        """Display all clusters along with their names."""
        from ase.gui.gui import GUI
        from ase.gui.images import Images
        figures = self.cluster_mng.get_figures()
        images = Images()
        images.initialize(figures)
        gui = GUI(images, expr='')
        gui.show_name = True
        gui.run()

    def get_all_templates(self):
        """
        Return a list with all template atoms
        """
        return self.template_atoms.get_all_templates()

    def view_templates(self):
        """
        Display all templates in the ASE GUi
        """
        from ase.visualize import view
        view(self.get_all_templates())

    def _check_first_elements(self):
        basis_elements = self.basis_elements
        num_basis = self.num_basis
        # This condition can be relaxed in the future
        first_elements = []
        for elements in basis_elements:
            first_elements.append(elements[0])
        if len(set(first_elements)) != num_basis:
            raise ValueError("First element of different basis should not be the same.")

    def save(self, filename):
        """Write Setting object to a file in JSON format.

        Parameters:

        filename: str
            Name of the file to store the necessary settings to initialize
            the Cluster Expansion calculations.
        """

        import json
        with open(filename, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=2)


def to_3x3_matrix(size):
    if size is None:
        return None
    if isinstance(size, np.ndarray):
        size = size.tolist()

    # Is already a matrix
    if np.array(size).shape == (3, 3):
        return size

    if np.array(size).shape == (3,):
        return np.diag(size).tolist()

    raise ValueError(f"Cannot convert passed array {size} to 3x3 matrix")
