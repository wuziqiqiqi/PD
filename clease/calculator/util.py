import json
import os
from clease import CorrFunction
from clease.calculator import Clease
from clease.tools import nested_list2str


def attach_calculator(setting=None, atoms=None, eci={},
                      fname_prefix=None, load=True):
    """Utility function for an efficient initialization of large cells.

    Parameters:

    setting: `ClusterExpansionSetting` object

    eci: dict
        Dictionary containing cluster names and their ECI values

    atoms: Atoms object
        Atoms object for MC simulations.

    fname_prefix: str
        Prefix of the file name for backing up cluster info in .json format.
        The file name will append '_cluster_info_{size_of_the_cell}.json'.

    load: bool
        Load cluster info if possible (the file with a name specified using
        the fname_prefix is present).
    """
    cf_names = list(eci.keys())
    init_cf = CorrFunction(setting).get_cf_by_names(setting.atoms, cf_names)

    template_uid = setting.template_atoms.get_uid_matching_atoms(
        atoms=atoms, generate_template=True)
    setting.prepare_new_active_template(template_uid)

    size_str = nested_list2str(setting.size)

    fname = '.'
    if fname_prefix is not None:
        fname = fname_prefix + '_cluster_info_{}.json'.format(size_str)

    loaded_info = False
    if load and os.path.exists(fname) and fname_prefix is not None:
        with open(fname, 'r') as infile:
            data = json.load(infile)
        setting.cluster_list = data['cluster_list']
        setting.trans_matrix = data['trans_matrix']
        loaded_info = True
    else:
        setting.create_cluster_list_and_trans_matrix()

    data = {'cluster_list': setting.cluster_list,
            'trans_matrix': setting.trans_matrix,
            'size': setting.size,
            'setting': setting.kwargs}

    if fname_prefix is not None and not loaded_info:
        save_info(fname, data)

    atoms = setting.atoms.copy()

    calc = Clease(setting, eci=eci, init_cf=init_cf)
    atoms.set_calculator(calc)
    return atoms


def save_info(fname, data):
    with open(fname, 'w') as outfile:
        json.dump(data, outfile)
