"""A collection of miscellaneous functions used for Cluster Expansion."""
from itertools import (
    permutations, combinations, product,
    filterfalse, chain
)
import numpy as np
from collections.abc import Iterable
from random import sample
from ase.db import connect
from clease import _logger
from scipy.spatial import cKDTree as KDTree


class ApproxEqualityList(object):
    """
    Wrapper around a list which implements a new comparison operator. If two
    items in the list is equal within a given tolerance, the items are
    considered to be equal.

    Parameters:

    array: list
        List that is wrapped

    tol: float
        Toleracnce for comparison check
    """

    def __init__(self, array, tol=1E-5):
        self.array = array
        self.tol = tol

    def __lt__(self, other):
        for x, y in zip(self.array, other.array):
            if x - y < -self.tol:
                return True
            elif x - y > self.tol:
                return False
        return False


def index_by_position(atoms):
    """Set atomic indices by its position."""
    # add zero to avoid negative zeros
    tags = atoms.get_positions() + 0
    tags = tags.tolist()
    tags = [ApproxEqualityList(x) for x in tags]
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


def shift(array):
    ref = array[-1]
    array[1:] = array[:-1]
    array[0] = ref
    return array


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
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


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
            ('name', '!=', 'template')]


def get_all_internal_distances(atoms, max_dist, ref_indices):
    """Obtain all internal distances of the passed atoms object and return a
       Numpy array containing all the distances sorted in an ascending order.
    """
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


def reconfigure(setting, select_cond=None):
    from clease import CorrFunction
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


def trans_matrix_index2tags(trans_matrix, tagged_atoms, indices=None):
    """
    Convert from indices to tags

    Parameters:

    trans_matrix: list of dict
        Original translation matrix

    tagged_atoms: Atoms
        Atoms with a tag that should be used instead of the
        index

    indices: list of int
        Atom indices corresponding to each row in trans_matrix.
        If None, it is assumed that len(trans_matrix) == len(tagged_atoms)
        and each row in trans_matrix corresponds to the atom with the same
        index in tagged_atoms.
    """
    unique_tags = sorted(list(set(atom.tag for atom in tagged_atoms)))

    if indices is None:
        indices = list(range(len(trans_matrix)))

    # Make sure we have a continuous series of tags
    assert len(unique_tags) == max(unique_tags) + 1

    new_tm = [{} for _ in range(len(unique_tags))]
    used_tags = [False for _ in range(len(unique_tags))]

    for i, row in enumerate(trans_matrix):
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


def list2str(array):
    return "-".join(str(x) for x in array)


def factorize(n):
    while n > 1:
        for i in range(2, n + 1):
            if n % i == 0:
                n = int(n/i)
                yield i
                break


def count_atoms(atoms):
    """
    Return dictionary with the number of items of each species
    """
    count = {}
    for s in atoms.symbols:
        count[s] = count.get(s, 0) + 1
    return count


def all_integer_transform_matrices_given_diag(diag):
    rng1 = range(0, diag[0]+1)
    rng2 = rng1
    rng3 = range(0, diag[1]+1)
    for off_diag in product(rng1, rng2, rng3):
        yield np.array(
            [[diag[0], off_diag[0], off_diag[1]],
             [0, diag[1], off_diag[2]],
             [0, 0, diag[2]]]
        )


def all_integer_transform_matrices_per_diag(n):
    """
    Yield all the integer transform matrices
    """
    diags = filterfalse(lambda x: x[0]*x[1]*x[2] != n,
                        product(range(n+1), repeat=3))

    for d in diags:
        yield all_integer_transform_matrices_given_diag(d)


def all_integer_transform_matrices(n):
    return chain(*all_integer_transform_matrices_per_diag(n))


def rotate_cells(cell, target_cell):
    """
    Rotate the cells such that one vector is parallel.
    And the other vector lies in a given plane
    """
    dot_prod_cell = cell.dot(cell.T)
    dot_prod_target_cell = target_cell.dot(target_cell.T)

    # Calculate unit vector cells
    uvec_cell = np.zeros_like(cell)
    uvec_target_cell = np.zeros_like(target_cell)
    for i in range(3):
        uvec_cell[i, :] = cell[i, :]/np.sqrt(dot_prod_cell[i, i])
        uvec_target_cell[i, :] = target_cell[i, :] / \
            np.sqrt(dot_prod_target_cell[i, i])

    # Rotate one vector to be parallel
    v = np.cross(uvec_cell[0, :], uvec_target_cell[0, :])
    c = uvec_cell[0, :].dot(uvec_target_cell[0, :])
    v_cross = np.zeros((3, 3))
    v[0, 1] = -v[2]
    v[0, 2] = v[1]
    v[1, 0] = v[2]
    v[1, 2] = -v[0]
    v[2, 0] = -v[1]
    v[2, 1] = v[0]
    R = np.eye(3) + v_cross + v_cross.dot(v_cross)/(1 + c)
    target_cell = R.dot(target_cell)
    return target_cell


def species_chempot2eci(bf_list, species_chempot):
    """
    Convert chemical potentials given for species to their corresponding
    singlet ECI values.

    Parameters:

    bf_list: list
        List of basis function values for each species

    species_chempot: dict
        Dictionary containing the chemical potential for each species. Note
        that the chemical potential for one species can be set to 0.0 without
        any loss of generality. Hence, the species_chempot should contain
        chemical potentials for all elements except one.
    """
    if len(species_chempot) != len(bf_list):
        msg = 'Inconsistent number of chemical potentials. Basis functions\n'
        msg += '{}. Passed chemical potentials {}'.format(bf_list,
                                                          species_chempot)
        raise ValueError(msg)

    n = len(species_chempot)
    mat = np.zeros((n, n))
    rhs = np.zeros(n)
    row = 0
    for sp, mu in species_chempot.items():
        for col, bf in enumerate(bf_list):
            mat[row, col] = bf[sp]
        rhs[row] = mu
        row += 1
    try:
        eci_chem_pot = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        inv_mat = np.linalg.pinv(mat)
        eci_chem_pot = inv_mat.dot(rhs)
    eci_chem_pot = eci_chem_pot.tolist()
    eci_dct = {'c1_{}'.format(i): v for i, v in enumerate(eci_chem_pot)}
    return eci_dct


def bf2matrix(bfs):
    """
    Convert a list of basis functions to a matrix. Each column represents
    a species and each row represent a basis function.

    Parameter:

    bfs: list of dict
        List of dictionaries containing the basis function values
    """
    nrows = len(bfs)
    ncols = len(bfs[0])
    mat = np.zeros((nrows, ncols))
    keys = sorted(list(bfs[0].keys()))
    for i, bf in enumerate(bfs):
        for j, symb in enumerate(keys):
            mat[i, j] = bf[symb]
    return mat


def rate_bf_subsets(elems, bfs):
    """
    Rate different combinations of basis function according to how
    well it is able to distinguish the elements in the basis.

    Example:
    bfs = [{'Li': 1.0, 'X': 0.0, 'V': 0.0, 'F': 0.0},
           {'Li': 0.0, 'X': 1.0, 'V': 0.0, 'F': 0.0},
           {'Li': 0.0, 'X': 0.0, 'V': 1.0, 'F': 0.0}]

    If one wants basis functions for the triplet [Li, X, F] we need two
    basis functions. The following combinations are possible
    (bfs[0], bfs[1]), (bfs[0], bfs[2]), (bfs[1], bfs[2])

    The score is defined as the sum of the absolute value of the difference
    between the basis function value for the selected symbols. The score for
    the first combinations is thus

    score1 = |bfs[0]['Li'] - bfs[0]['F']| + |bfs[0]['Li'] - bfs[0]['X']| +
             |bfs[0]['F'] - bfs[0]['X']|  + |bfs[1]['Li'] - bfs[1]['F']| +
             |bfs[1]['Li'] - bfs[1]['X']| + |bfs[1]['F']  - bfs[1]['X']|

    Therefore,
    score1 = 1.0 + 1.0 + 0.0 + 0.0 + 1.0 + 1.0 = 4.0

    Parameter:

    bfs: list of dict
        List with dictionaries with the basis function values
    """
    score_and_comb = []
    for comb in combinations(range(len(bfs)), r=len(elems)-1):
        chosen = [{s: bfs[x][s] for s in elems} for x in comb]
        mat = bf2matrix(chosen)
        score = 0.0
        for row in range(mat.shape[0]):
            for i in product(range(mat.shape[1]), repeat=2):
                score += abs(mat[row, i[0]] - mat[row, i[1]])

        score_and_comb.append((score, list(comb)))
    score_and_comb.sort(reverse=True)
    return score_and_comb


def select_bf_subsets(basis_elems, bfs):
    """
    Select a subset of basis functions that best describes the basis.
    The best subset is the one that yields the highest sum of the scores for
    each sublattice. For definition of score see docstring
    of `rate_bf_subsets`

    Parameters:

    basis_elem: nested list of strings
        Basis elements (e.g. [[Li, V], [O]])

    bfs: list of dicts
        List of dictionaries holding the basis function values
        (e.g. [{'Li' : 1.0, 'V': 0.0, 'O': 0.0},
               {'Li': 0.0, 'O': 1.0, 'V': 0.0}])
    """
    rates = []
    for elems in basis_elems:
        rates.append(rate_bf_subsets(elems, bfs))

    # Find the combination of subset selections that gives the overall highest
    # score given that a basis function can be present in only one basis
    best_selection = []
    best_score = None
    for comb in product(*rates):
        total_score = sum(rate[0] for rate in comb)
        selection = [rate[1] for rate in comb]

        # Check that no basis function is selected twice
        selected_bfs = set()
        duplicates = False
        for s in selection:
            for bf in s:
                if bf in selected_bfs:
                    duplicates = True
                else:
                    selected_bfs.add(bf)

        # Add penalty to the ones that has duplicate CF functions. This way we
        # select a combination that has the same basis function in multiple
        # atomic basis if it is possible
        if duplicates:
            total_score -= 1000.0

        if best_score is None or total_score > best_score:
            best_score = total_score
            best_selection = selection
    return best_selection


def cname_lt(cname1, cname2):
    """
    Compare two cluster names to check if the first cluster name is
    smaller than (less than, or lt) the second cluster name. Since the cluster
    names take a form 'c#_d####_#', the prefix ('c#_d####') is evaluated as a
    string while the suffix ('#') is evaluated as an integer.

    Return `True` if cname1 < cname2 and `False` otherwise.
    """
    if not isinstance(cname1, str) and isinstance(cname2, str):
        raise TypeError('cnames should be strings.')

    if cname1 in ('c0', 'c1'):
        prefix1 = cname1
    else:
        prefix1 = cname1.rpartition("_")[0]

    if cname2 in ('c0', 'c1'):
        prefix2 = cname2
    else:
        prefix2 = cname2.rpartition("_")[0]

    if prefix1 < prefix2:
        return True
    elif prefix1 > prefix2:
        return False

    # Case where prefixes are the same.
    if cname1 in ('c0', 'c1'):
        suffix1 = 0
    else:
        suffix1 = int(cname1.rpartition("_")[-1])

    if cname2 in ('c0', 'c1'):
        suffix2 = 0
    else:
        suffix2 = int(cname2.rpartition("_")[-1])

    if suffix1 < suffix2:
        return True
    else:
        return False
