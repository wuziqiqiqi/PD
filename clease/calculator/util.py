from clease import CorrFunction, ClusterExpansionSettings
from clease.calculator import Clease
from ase import Atoms
from typing import Dict


def attach_calculator(settings: ClusterExpansionSettings,
                      atoms: Atoms, eci: Dict[str, float] = {}) -> Atoms:
    """Utility function for an efficient initialization of large cells.

    :param settings: ClusterExpansionSettings object (e.g., CEBulk, CECyrstal)
    :param eci: Dictionary containing cluster names and their ECI values
    :param atoms: ASE Atoms object.
    """
    cf_names = list(eci.keys())
    init_cf = CorrFunction(settings).get_cf_by_names(settings.atoms, cf_names)

    template = settings.template_atoms.get_template_matching_atoms(
        atoms=atoms)
    settings.set_active_template(atoms=template)

    atoms = settings.atoms.copy()

    calc = Clease(settings, eci=eci, init_cf=init_cf)
    atoms.set_calculator(calc)
    return atoms


def get_ce_energy(settings: ClusterExpansionSettings, atoms: Atoms,
                  eci: Dict[str, float]) -> float:
    """Get energy of the ASE Atoms object based on given ECI values.

    :param settings: ClusterExpansionSettings object (e.g., CEBulk, CECyrstal)
    :param atoms: ASE Atoms object representing the considered atomic
        arrangement
    :param eci: ECI values to be used to calculate the energy
    """
    # temp atoms to avoid making unexpected changes to passed atoms
    temp_atoms = atoms.copy()
    calc = Clease(settings, eci=eci)
    temp_atoms.set_calculator(calc)
    energy = temp_atoms.get_potential_energy()
    return energy
