"""A collection of miscellaneous functions used for Cluster Expansion."""
from itertools import permutations, combinations, product
import numpy as np
import collections
from random import sample
from ase.db import connect
import json
from clease import _logger
from ase.geometry import wrap_positions
from scipy.spatial import cKDTree as KDTree


def index_by_position(atoms):
    """Set atomic indices by its position."""
    # add zero to avoid negative zeros
    tags = atoms.get_positions().round(decimals=6) + 0
    tags = tags.tolist()
    deco = sorted([(tag, i) for i, tag in enumerate(tags)])
    indices = [i for tag, i in deco]
    return indices


def sort_by_position(atoms):
    """Sort atoms by its position."""
    # Return a new Atoms object with sorted atomic order.
    # The default is to order according to chemical symbols,
    # but if *tags* is not None, it will be used instead.
    # A stable sorting algorithm is used.
    indices = index_by_position(atoms)
    return atoms[indices]


def wrap_and_sort_by_position(atoms):
    """Wrap and sort atoms by their positions."""
    atoms.wrap()
    atoms = sort_by_position(atoms)
    return atoms


def create_cluster(atoms, indices):
    """Create a cluster centered in the unit cell."""
    return atoms[indices]


def shift(array):
    ref = array[-1]
    array[1:] = array[:-1]
    array[0] = ref
    return array


def distances_and_angles(atoms, ref_indx, float_obj_angle):
    """Get sorted internal angles of a."""
    indices = [a.index for a in atoms if a.index != ref_indx]
    if len(atoms) < 2:
        raise ValueError("distances and angles cannot be called for"
                         "{} body clusters".format(len(atoms)))
    if len(atoms) == 2:
        dist = atoms.get_distance(ref_indx, indices[0])
        classifier = distance_string(atoms.info["distances"], dist)
        return [classifier]

    angles = []
    dists = []

    for comb in combinations(indices, r=2):
        angle = atoms.get_angle(comb[0], ref_indx, comb[1])
        ang_classifier = float_obj_angle.get(angle)
        angles.append(ang_classifier)

    dists = atoms.get_distances(ref_indx, indices, mic=True)
    dists = sorted(dists.tolist(), reverse=True)
    dists = [distance_string(atoms.info["distances"], d) for d in dists]
    return dists + sorted(angles, reverse=True)


def get_cluster_descriptor(cluster, float_obj_angle):
    """Create a unique descriptor for each cluster."""
    dist_ang_tuples = []
    for ref_indx in range(len(cluster)):
        dist_ang_list = distances_and_angles(cluster, ref_indx,
                                             float_obj_angle)
        dist_ang_tuples.append(dist_ang_list)
    return dist_ang_tuples


def sort_by_internal_distances(atoms, indices, float_obj_ang):
    """Sort the indices according to the distance to the other elements."""
    if len(indices) <= 1:
        return list(range(len(indices))), "point"

    cluster = create_cluster(atoms, indices)
    if len(indices) == 2:
        dist_ang = get_cluster_descriptor(cluster, float_obj_ang)
        order = list(range(len(indices)))
        eq_sites = [(0, 1)]
        descr = "{}_0".format(dist_ang[0][0])
        return order, eq_sites, descr

    dist_ang = get_cluster_descriptor(cluster, float_obj_ang)
    sort_order = [ind for _, ind in sorted(zip(dist_ang, range(len(indices))))]
    dist_ang.sort()
    equivalent_sites = [[i] for i in range(len(indices))]
    site_types = [i for i in range(len(indices))]
    for i in range(len(sort_order)):
        for j in range(i + 1, len(sort_order)):
            if dist_ang[i] == dist_ang[j]:
                if site_types[j] > i:
                    # This site has not been assigned to another category yet
                    site_types[j] = i
                st = site_types[j]
                if j not in equivalent_sites[st]:
                    equivalent_sites[st].append(j)

    # Remove empty lists from equivalent_sites
    equivalent_sites = [entry for entry in equivalent_sites if len(entry) > 1]

    # Create a string descriptor of the clusters
    dist_ang_strings = []
    for item in dist_ang:
        strings = [str(x) for x in item]
        dist_ang_strings.append("_".join(strings))
    string_description = "-".join(dist_ang_strings)
    return sort_order, equivalent_sites, string_description


def ndarray2list(data):
    """
    Convert nested lists of a combination of lists and numpy arrays
    to list of lists
    """
    if not isinstance(data, list) and not isinstance(data, np.ndarray):
        return data

    data = list(data)
    for i in range(len(data)):
        data[i] = ndarray2list(data[i])
    return list(data)


def dec_string(deco, equiv_sites):
    """Create the decoration string based on equiv sites."""
    equiv_dec = sorted(equivalent_deco(deco, equiv_sites))
    return ''.join(str(i) for i in equiv_dec[0])


def equivalent_deco(deco, equiv_sites):
    """Generate equivalent decoration numbers based on equivalent sites."""
    if not equiv_sites:
        return [deco]

    perm = []
    for equiv in equiv_sites:
        perm.append(list(permutations(equiv)))

    equiv_deco = []
    for comb in product(*perm):
        order = []
        for item in comb:
            order += list(item)

        orig_order = list(range(len(deco)))
        for i, srt_indx in enumerate(sorted(order)):
            orig_order[srt_indx] = order[i]
        equiv_deco.append([deco[indx] for indx in orig_order])

    unique_deco = []
    for eq_dec in equiv_deco:
        if eq_dec not in unique_deco:
            unique_deco.append(eq_dec)
    return unique_deco


def flatten(x):
    """Flatten list."""
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def get_unique_name(size, max_dia, fam_id):
    """Get unique cluster names."""
    name = "c{}_{}_{}".format(size, max_dia, fam_id)
    return name


def nested_array2list(array):
    """Convert a nested array/tuple to a nested list."""
    if isinstance(array, np.ndarray):
        array = array.tolist()
    else:
        array = list(array)
    try:
        for i in range(len(array)):
            if isinstance(array[i], np.ndarray):
                array[i] = array[i].tolist()
            else:
                array[i] = list(array[i])
    except TypeError:
        pass
    return array


def update_db(uid_initial=None, final_struct=None, db_name=None,
              custom_kvp_init={}, custom_kvp_final={}):
    """Update the database.

    Parameters:

    uid_initial: int
        entry ID of the initial structure in the database

    final_struct: Atoms
        Atoms object with the final structure with a physical
        quantity that needs to be modeled (e.g., DFT energy)

    db_name: str
        Database name

    custom_kvp_init: dict (optional)
        If desired, one can pass additional key-value-pairs for the
        entry containing the initial structure

    custom_kvp_final: dict (optional)
        If desired, one can pass additional key-value-pairs for the
        entry containing the final structure
    """
    from ase.db import connect
    db = connect(db_name)

    init_row = db.get(id=uid_initial)

    # Check if a final structure already exits
    name = init_row.name
    select_cond = [('name', '=', name), ('struct_type', '=', 'final')]
    exist = sum(1 for row in db.select(select_cond))
    if exist >= 1:
        _logger("A structure with 'name'={} and 'struct_type'=final "
                "already exits in DB".format(name))
        return

    # Write the final structure to database
    kvp_final = {'struct_type': 'final', 'name': name}
    kvp_final.update(custom_kvp_final)
    uid_final = db.write(final_struct, key_value_pairs=kvp_final)

    kvp_update_init = {
        'converged': True,
        'started': '',
        'queued': '',
    }
    if kvp_final['struct_type'] == 'final':
        kvp_update_init['final_struct_id'] = uid_final
    kvp_update_init.update(custom_kvp_init)

    # Update info for the initial structure
    db.update(uid_initial, **kvp_update_init)


def exclude_information_entries():
    """Return selection condition to exlcude all entries in the database that
       only contain information about the clusters.
    """
    return [('name', '!=', 'primitive_cell'),
            ('name', '!=', 'template'),
            ('name', '!=', 'float_classification')]


def get_all_internal_distances(atoms, max_dist, ref_indices):
    """Obtain all internal distances of the passed atoms object and return a
       Numpy array containing all the distances sorted in an ascending order.
    """
    from scipy.spatial import cKDTree as KDTree

    tree = KDTree(atoms.get_positions())
    distances = []
    for ind in ref_indices:
        indices = tree.query_ball_point(atoms[ind].position, max_dist)
        dists = atoms.get_distances(ind, indices)
        for d in dists:
            if np.any(np.abs(np.array(distances) - d) < 1E-6):
                continue
            distances.append(d)
    distances = sorted(distances)

    # Make sure that the first element is 0
    assert distances[0] < 1E-6
    return np.array(distances[1:])


def distance_string(distance_array, distance):
    """Provide a name of the passed distance in terms of the nearest neighbor
       based on the internal distances in an array."""
    indx = np.argmin(np.abs(distance_array - distance))
    assert abs(distance_array[indx] - distance) < 1E-6

    if indx < 9:
        return "0{}nn".format(indx+1)
    return "{}nn".format(indx+1)


def reconfigure(setting, select_cond=None):
    from clease import CorrFunction
    setting.reconfigure_settings()
    CorrFunction(setting).reconfigure_db_entries(select_cond)


def split_dataset(X, y, nsplits=10):
    """Split the dataset such that it can be used for k-fold
        cross validation."""
    from random import shuffle
    indices = list(range(len(y)))
    shuffle(indices)
    partitions = []
    num_validation = int(len(y)/nsplits)

    if num_validation < 1:
        num_validation = 1
    for i in range(nsplits):
        start = i*num_validation
        end = (i+1)*num_validation
        if i == nsplits-1:
            indx = indices[start:]
        else:
            indx = indices[start:end]
        mask = np.zeros(len(y), dtype=np.uint8)
        mask[indx] = 1
        data = {
            "train_X": X[mask == 0, :],
            "train_y": y[mask == 0],
            "validate_X": X[mask == 1, :],
            "validate_y": y[mask == 1]
        }
        partitions.append(data)
    return partitions


def random_validation_set(num=10, select_cond=None, db_name=None):
    """
    Construct a random test set.

    Parameters:

    num: int
        Number of datapoints to include in the test set
    select_cond: list
        Select condition to be used to select items from
        the database. If not given, it will be struct_type='initial',
        converged=True
    db_name str:
        Name of the database
    """

    db = connect(db_name)
    if select_cond is None:
        select_cond = [('struct_type', '=', 'initial'),
                       ('converged', '=', True)]

    if db_name is None:
        raise ValueError("No database provided!")

    all_ids = []
    for row in db.select(select_cond):
        all_ids.append(row.id)
    return sample(all_ids, num)


def exclude_ids(ids):
    """
    Construct a select condition based on the ids passed.

    Parameters:

    ids: list of int
        List of IDs
    """
    return [("id", "!=", x) for x in ids]


def load_settings(fname):
    """Load Cluster Expansion settings from a JSON file.

    Parameters:

    fname: str
        JSON file with the settings information
    """
    with open(fname, 'r') as infile:
        info = json.load(infile)

    if info['classtype'] == 'CEBulk':
        from clease import CEBulk
        return CEBulk.load(fname)
    elif info['classtype'] == 'CECrystal':
        from clease import CECrystal
        return CECrystal.load(fname)

    allowed_class_types = ['CEBulk', 'CECrystal']
    raise ValueError("Could not find matching ClusterExpansionSetting type. "
                     "Ensure that the classtype field in the JSON file is one "
                     "of {}".format(allowed_class_types))


def symbols2integer(basis_functions):
    """Convert each symbol in the basis function
       to a unique integer.

    Parameters:

    basis_functions: list of dict
        The basis function dictionary from ClusterExpansionSetting
    """
    symb_id = {}
    for i, symb in enumerate(basis_functions[0].keys()):
        symb_id[symb] = i
    return symb_id


def bf2npyarray(basis_functions, symb_id):
    """Convert the basis function dictionary to a 2D
        numpy array

    Parameters:

    basis_function: list of dict
        Basis function dictionary from ClusterExpansionSetting

    symb_id: dict
        Dictionary of symbol integer (see symbols2integer)
    """
    bf_npy = np.zeros((len(basis_functions), len(symb_id)))

    for i, bf in enumerate(basis_functions):
        for k, v in bf.items():
            bf_npy[i, symb_id[k]] = v
    return bf_npy


def nested_list2str(nested_list):
    """Convert a nested list to string."""
    return 'x'.join(','.join(str(x) for x in item) for item in nested_list)


def str2nested_list(string):
    """Convert string to nested list."""
    return [list(map(lambda x: int(x), item.split(',')))
            for item in string.split('x')]


# def close_to_cubic_supercell(atoms, zero_cutoff=0.1):
#     """
#     Create a close to cubic supercell.

#     Parameters:

#     atoms: Atoms
#         Cell to be used for construction

#     zero_cutoff: float
#         Value below this value will be considered as zero when the
#         scaling factor is computed
#     """
#     cell = atoms.get_cell()
#     a = np.linalg.det(cell)**(1.0/3.0)
#     inv_cell = np.linalg.inv(cell)
#     scale = 1.0/inv_cell[np.abs(inv_cell)*a > zero_cutoff]
#     scale = np.round(scale).astype(np.int32)
#     min_gcd = min([gcd(scale[0], scale[i]) for i in range(len(scale))])
#     scale = np.true_divide(scale, min_gcd)
#     scale = min_gcd*np.max(scale)
#     integer_matrix = np.round(inv_cell*scale).astype(np.int32)

#     if np.linalg.det(integer_matrix) < 0:
#         integer_matrix *= -1

#     sc = make_supercell(atoms, integer_matrix)
#     sc = wrap_and_sort_by_position(sc)

#     # We need to tag the atoms
#     sc_pos = sc.get_positions()
#     sc_pos = wrap_positions(sc_pos, atoms.get_cell())

#     tree = KDTree(atoms.get_positions())
#     dists, tags = tree.query(sc_pos)
#     assert np.allclose(dists, 0.0)
#     for i, tag in enumerate(tags):
#         sc[i].tag = tag
#     return sc


def min_distance_from_facet(x, cell):
    """
    Calculate the minimum distance from a point to the cell facet.

    Parameters:

    x: np.array
        Position from which to calculate the minimum distance

    cell: Cell
        Cell of an Atoms object
    """
    dists = []

    for plane in combinations([0, 1, 2], r=2):
        # Unit normal vector
        n = np.cross(cell[plane[0], :], cell[plane[1], :])
        n /= np.sqrt(n.dot(n))

        # Plane with origin in it
        dist = np.abs(n.dot(x))
        dists.append(dist)

        # Opposite facet
        remaining = list(set([0, 1, 2]) - set(plane))[0]
        vec = cell[remaining, :]
        dist = np.abs(n.dot(x - vec))
        dists.append(dist)
    return min(dists)


def trans_matrix_index2tags(tm, tagged_atoms, indices=None):
    """
    Convert from indices to tags

    Parameters:

    tm: list of dict
        Original translation matrix

    tagged_atoms: Atoms
        Atoms with a tag that should be used instead of the
        index

    indices: list of int
        Atom indices corresponding to each row in tm. If None,
        it is assumed that len(tm) == len(tagged_atoms) and each
        row in tm corresponds to the atom with the same index in
        tagged_atoms.
    """
    unique_tags = sorted(list(set(atom.tag for atom in tagged_atoms)))

    if indices is None:
        indices = list(range(len(tm)))

    # Make sure we have a continuous series of tags
    assert len(unique_tags) == max(unique_tags) + 1

    new_tm = [{} for _ in range(len(unique_tags))]
    used_tags = [False for _ in range(len(unique_tags))]

    for i, row in enumerate(tm):
        tag = tagged_atoms[indices[i]].tag
        if used_tags[tag]:
            continue
        new_row = {tagged_atoms[k].tag: tagged_atoms[v].tag
                   for k, v in row.items()}
        used_tags[tag] = True
        new_tm[tag] = new_row
    return new_tm


def indices2tags(supercell, clusters):
    for cluster in clusters:
        for i, figure in enumerate(cluster):
            cluster[i] = [int(supercell[x].tag) for x in figure]
    return clusters

