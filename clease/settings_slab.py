from clease import (
    ClusterExpansionSettings, Concentration
)
from ase.build import surface
from ase import Atoms
from typing import Union, Tuple, Optional
import numpy as np


def CESlab(conventional_cell: Union[Atoms, str],
           miller: Tuple[int], concentration: Concentration,
           size: Optional[Tuple[int]] = (1, 1, 1),
           max_cluster_size: Optional[int] = 4,
           max_cluster_dia: Optional[Tuple[float]] = (5.0, 5.0, 5.0),
           supercell_factor: Optional[int] = 27,
           db_name: Optional[str] = 'clease.db'
           ) -> ClusterExpansionSettings:
    """
    :param conventional_cell:
        Bulk lattice structure. Note that the unit-cell
        must be the conventional cell - not the primitive cell. One can also
        give the chemical symbol as a string, in which case the correct bulk
        lattice will be generated automatically.

    :param miller:
        Surface normal in Miller indices (h,k,l).

    :param concentration:
        Class for restricting the concentrations

    :param size:
        Size of the simulations cell. The third number represents the number of
        layers. The two first are repetitions of the in-plane unit vectors

    :param max_cluster_size:
        Maximum number of atoms in cluster

    :param max_cluster_dia:
        Maximum cluster diameters

    :param supercell_factor:
        Maximum number of unit cells that can be used to create a supercell

    :param db_name:
        Name of the database where structures should be placed
    """

    for b in concentration.basis_elements:
        if 'X' not in b:
            raise ValueError("Slab calculation requires that X is present "
                             "in all basis")

    prim = get_prim_slab_cell(conventional_cell, miller)

    # Slab should always have one cell vector along the z-axis
    setting = ClusterExpansionSettings(
        prim, concentration, size=size,
        max_cluster_dia=max_cluster_dia,
        max_cluster_size=max_cluster_size,
        supercell_factor=supercell_factor,
        db_name=db_name
    )

    dict_rep = conventional_cell.todict()
    for k, v in dict_rep.items():
        if isinstance(v, np.ndarray):
            dict_rep[k] = v.tolist()

    setting.kwargs.update(
        {
            'factory': 'CESlab',
            'miller': miller,
            'conventional_cell': dict_rep,
            'size': size
        }
    )
    return setting


def get_prim_slab_cell(conventional_cell: Union[Atoms, str],
                       miller: Tuple[int]) -> Atoms:
    """
    Returns the primitive cell used for slab CE

    :param conventional_cell:
        Bulk lattice structure. Note that the unit-cell
        must be the conventional cell - not the primitive cell. One can also
        give the chemical symbol as a string, in which case the correct bulk
        lattice will be generated automatically.

    :param miller:
        Surface normal in Miller indices (h,k,l).
    """
    prim = surface(conventional_cell, miller, 1, periodic=True)
    return prim


def add_vacuum_layers(atoms: Atoms, thickness: float) -> Atoms:
    """
    Add vacuum layers to the slab

    :param atoms: Atoms object representing the slab

    :param thickness: Thickness of the vacuum layer
    """
    vacuum = atoms.copy()
    for atom in vacuum:
        atom.symbol = 'X'

    dz = atoms.get_cell()[2, 2]
    vacuum.translate([0, 0, dz])

    num_cells = int(thickness/dz) + 1
    vacuum *= (1, 1, num_cells)

    atoms += vacuum
    cell = atoms.get_cell()
    cell[2, 2] += num_cells*dz
    atoms.set_cell(cell)
    return atoms
