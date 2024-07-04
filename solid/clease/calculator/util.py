from typing import Dict, Optional
from ase import Atoms
from clease.settings import ClusterExpansionSettings
from clease.tools import wrap_and_sort_by_position
from .clease import Clease


def attach_calculator(
    settings: ClusterExpansionSettings,
    atoms: Atoms,
    eci: Dict[str, float] = None,
    num_threads: Optional[int] = None,
) -> Atoms:
    """Utility function for an efficient initialization of large cells. Will set the atoms
    object as the active template in the settings.

    :param settings: ClusterExpansionSettings object (e.g., CEBulk, CECyrstal)
    :param eci: Dictionary containing cluster names and their ECI values
    :param atoms: ASE Atoms object.
    :param num_threads: Number of threads to use when updating CFs. Requires CLEASE to
        be compiled with OpenMP if the value is different from 1.
    """
    if eci is None:
        eci = {}

    # The settings object will complain if we cannot coerce the atoms object into a template
    # object.
    settings.set_active_template(atoms=atoms)

    wrapped = wrap_and_sort_by_position(atoms)
    atoms_with_calc = settings.atoms.copy()

    calc = Clease(settings, eci=eci)
    atoms_with_calc.symbols = wrapped.symbols
    atoms_with_calc.calc = calc
    if num_threads is not None:
        calc.set_num_threads(num_threads)

    return atoms_with_calc


def get_ce_energy(settings: ClusterExpansionSettings, atoms: Atoms, eci: Dict[str, float]) -> float:
    """Get energy of the ASE Atoms object based on given ECI values.

    :param settings: ClusterExpansionSettings object (e.g., CEBulk, CECyrstal)
    :param atoms: ASE Atoms object representing the considered atomic
        arrangement
    :param eci: ECI values to be used to calculate the energy
    """
    # temp atoms to avoid making unexpected changes to passed atoms
    temp_atoms = atoms.copy()
    calc = Clease(settings, eci=eci)
    temp_atoms.calc = calc
    energy = temp_atoms.get_potential_energy()
    return energy
