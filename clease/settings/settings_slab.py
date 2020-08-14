from typing import Union, Tuple, Optional
import numpy as np
from ase import Atoms
from ase.build import surface, make_supercell
from ase.geometry import get_layers

from .settings import ClusterExpansionSettings
from . import Concentration

__all__ = ('CESlab',)


def CESlab(conventional_cell: Union[Atoms, str],
           miller: Tuple[int],
           concentration: Concentration,
           size: Optional[Tuple[int]] = (1, 1, 1),
           max_cluster_size: Optional[int] = 4,
           max_cluster_dia: Optional[Tuple[float]] = (5.0, 5.0, 5.0),
           supercell_factor: Optional[int] = 27,
           db_name: Optional[str] = 'clease.db') -> ClusterExpansionSettings:
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
            raise ValueError("Slab calculation requires that X is present in all basis")

    prim = get_prim_slab_cell(conventional_cell, miller)

    # Slab should always have one cell vector along the z-axis
    settings = ClusterExpansionSettings(prim,
                                        concentration,
                                        size=size,
                                        max_cluster_dia=max_cluster_dia,
                                        max_cluster_size=max_cluster_size,
                                        supercell_factor=supercell_factor,
                                        db_name=db_name)

    dict_rep = conventional_cell.todict()
    for k, v in dict_rep.items():
        if isinstance(v, np.ndarray):
            dict_rep[k] = v.tolist()

    settings.kwargs.update({
        'factory': 'CESlab',
        'miller': miller,
        'conventional_cell': dict_rep,
        'size': size
    })
    return settings


def get_prim_slab_cell(conventional_cell: Union[Atoms, str], miller: Tuple[int]) -> Atoms:
    """
    Returns the primitive cell used for slab CE

    :param conventional_cell:
        Bulk lattice structure. Note that the unit cell must be a conventional
        cell - not the primitive cell. One can also give the chemical symbol
        as a string, in which case the correct bulk lattice will be generated
        automatically.

    :param miller:
        Surface normal in Miller indices (h,k,l).
    """
    prim = surface(conventional_cell, miller, 1, periodic=True)
    return prim


def add_vacuum_layers(atoms: Atoms, prim: Atoms, thickness: float) -> Atoms:
    """
    Add vacuum layers to the slab

    :param atoms:
        ASE Atoms object representing the slab

    :param thickness:
        Thickness of the vacuum layer
    """
    # construct one layer of vacuum filled with vacancies
    prim_xy = prim.cell[:3]
    atoms_xy = atoms.cell[:3]
    P = atoms_xy.dot(np.linalg.inv(prim_xy))[:2].astype(int)
    P = np.vstack((P, [0, 0, 1]))
    vacuum = make_supercell(prim, P)
    for atom in vacuum:
        atom.symbol = 'X'

    dz = atoms.get_cell()[2, 2]
    vacuum.translate([0, 0, dz])

    one_layer_thickness = vacuum.get_cell()[2, 2]
    # upside-down division to get ceiling division
    num_vac_layers = int(-(-thickness // one_layer_thickness))
    vacuum *= (1, 1, num_vac_layers)

    atoms += vacuum
    atoms.cell[2, 2] += vacuum.cell[2, 2]
    return atoms


def remove_vacuum_layers(atoms: Atoms) -> Atoms:
    """
    Remove vacuum layers from the slab.

    :param atoms:
        ASE Atoms object representing the slab with vacuum
    """
    atoms = atoms.copy()
    tags, levels = get_layers(atoms, miller=(0, 0, 1))
    num_total_layers = np.amax(tags, axis=None) + 1
    vac_layers = []

    for layer in range(num_total_layers):
        symbols = [atoms[i].symbol for i, tag in enumerate(tags) if tag == layer]
        if set(symbols) == {'X'}:
            vac_layers.append(layer)

    del atoms[[atoms[i].index for i, tag in enumerate(tags) if tag in vac_layers]]

    num_vac_layers = len(vac_layers)
    atoms.cell[2, 2] *= (num_total_layers - num_vac_layers) / num_total_layers

    return atoms
