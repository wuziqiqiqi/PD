import json
import os
from clease import CorrFunction
from clease.calculator import Clease
from clease.tools import nested_list2str


def attach_calculator(setting=None, atoms=None, eci={}):
    """Utility function for an efficient initialization of large cells.

    Parameters:

    setting: `ClusterExpansionSetting` object

    eci: dict
        Dictionary containing cluster names and their ECI values

    atoms: Atoms object
        Atoms object for MC simulations.

    """
    cf_names = list(eci.keys())
    init_cf = CorrFunction(setting).get_cf_by_names(setting.atoms, cf_names)

    template = setting.template_atoms.get_template_matching_atoms(
        atoms=atoms)
    setting.set_active_template(atoms=template)

    atoms = setting.atoms.copy()

    calc = Clease(setting, eci=eci, init_cf=init_cf)
    atoms.set_calculator(calc)
    return atoms


def save_info(fname, data):
    with open(fname, 'w') as outfile:
        json.dump(data, outfile)
