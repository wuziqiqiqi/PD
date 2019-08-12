import json
import os


def attach_calculator(setting=None, atoms=None, eci={}, prefix=None,
                      load=True):
    """
    Utility function for efficient initialisation of large cells

    Parameters:

    setting: `ClusterExpansionSetting`
        Settings object

    eci: dict
        Dictionary with the effective cluster interactions

    atoms: Atoms object
        Atoms object for MC simulations

    prefix: str
        Prefix for backup

    load: bool
        Load cluster info if possible
    """
    from clease import CorrFunction
    from clease.calculator import Clease
    from clease.tools import nested_list2str

    cf = CorrFunction(setting)
    init_cf = cf.get_cf(setting.atoms)

    template_uid = setting.template_atoms.get_uid_matching_atoms(
        atoms=atoms, generate_template=True)
    setting.prepare_new_active_template(template_uid)

    size_str = nested_list2str(setting.size)

    fname = '.'
    if prefix is not None:
        fname = prefix + 'cluster_info{}.json'.format(size_str)
    loaded_info = False
    if load and os.path.exists(fname) and prefix is not None:
        with open(fname, 'r') as infile:
            data = json.load(infile)
        setting.cluster_info = data['cluster_info']
        setting.trans_matrix = data['trans_matrix']
        loaded_info = True
    else:
        setting.create_cluster_info_and_trans_matrix()

    data = {
            'cluster_info': setting.cluster_info,
            'trans_matrix': setting.trans_matrix,
            'size': setting.size,
            'setting': setting.kwargs
        }

    if prefix is not None and not loaded_info:
        save_info(fname, data)

    atoms = setting.atoms.copy()

    calc = Clease(setting, cluster_name_eci=eci, init_cf=init_cf)
    atoms.set_calculator(calc)
    return atoms


def save_info(fname, data):
    with open(fname, 'w') as outfile:
        json.dump(data, outfile)
