from itertools import product
import pytest
import numpy as np
from ase.build import bulk
from ase.db import connect
from clease.tools import (min_distance_from_facet, factorize, all_integer_transform_matrices,
                          species_chempot2eci, bf2matrix, rate_bf_subsets, select_bf_subsets,
                          cname_lt, singlets2conc, aic, aicc, bic, get_extension,
                          add_file_extension, equivalent_deco, sort_cf_names, split_dataset,
                          common_cf_names, constraint_is_redundant, remove_redundant_constraints,
                          remove_redundant_equations)
from clease.basis_function import Polynomial


def test_equivalent_deco():
    predict_result = [{
        'deco': [1, 2, 3, 4],
        'equiv_site': [[0, 1, 2]],
        'result': [[1, 2, 3, 4], [1, 3, 2, 4], [2, 1, 3, 4], [2, 3, 1, 4], [3, 1, 2, 4],
                   [3, 2, 1, 4]]
    }, {
        'deco': [1, 2, 3, 4],
        'equiv_site': [[0, 3]],
        'result': [[1, 2, 3, 4], [4, 2, 3, 1]],
    }, {
        'deco': [1, 2, 3, 4],
        'equiv_site': [],
        'result': [[1, 2, 3, 4]]
    }]
    method_result = []
    for dict_list in predict_result:
        method_result.append(equivalent_deco(dict_list['deco'], dict_list['equiv_site']))

    for count, result_method in enumerate(method_result):
        assert predict_result[count]['result'] == result_method


def test_min_distance_from_facet():
    a = 4.0
    atoms = bulk('Al', crystalstructure='sc', a=a)

    x = [3.0, 3.5, 3.9]
    dist = min_distance_from_facet(x, atoms.get_cell())
    assert dist == pytest.approx(0.1)


def test_factorize():
    fact = sorted(list(factorize(10)))
    assert fact == [2, 5]

    fact = sorted(list(factorize(16)))
    assert fact == [2, 2, 2, 2]

    fact = sorted(list(factorize(24)))
    assert fact == [2, 2, 2, 3]


def test_all_int_matrices():
    arr = all_integer_transform_matrices(10)
    assert sum(1 for _ in arr) == 582


def test_species_chempot2eci():
    tests = [{
        'species': {
            'Au': 1.5
        },
        'bf_list': [{
            'Au': 1.0,
            'Cu': -1.0
        }],
        'expect': {
            'c1_0': 1.5
        }
    }, {
        'species': {
            'Au': 1.5,
            'Cu': 0.5
        },
        'bf_list': [{
            'Au': 0.3,
            'Cu': 1.2,
            'X': 3.0
        }, {
            'Au': -0.3,
            'Cu': 1.2,
            'X': -3.0
        }],
        'expect': {
            'c1_0': 65 / 24,
            'c1_1': -55 / 24
        }
    }]

    for i, test in enumerate(tests):
        eci = species_chempot2eci(test['bf_list'], test['species'])
        msg = 'Test #{} failed '.format(i)
        msg += 'Setup: {}'.format(test)
        for k, v in eci.items():
            assert v == pytest.approx(test['expect'][k]), msg


def test_bf2matrix():
    tests = [{
        'bf': [{
            'Al': 1.0,
            'Mg': -1.0
        }],
        'expect': np.array([[1.0, -1.0]])
    }, {
        'bf': [{
            'Li': 1.0,
            'O': 0.0,
            'X': 0.0
        }, {
            'Li': 0.0,
            'O': 1.0,
            'X': 0.0
        }],
        'expect': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    }]

    for test in tests:
        mat = bf2matrix(test['bf'])
        assert np.allclose(mat, test['expect'])


def test_subset_selection():
    tests = [{
        'bfs': [{
            'Al': 1.0,
            'Mg': 1.0,
            'Si': 0.0
        }, {
            'Al': 1.0,
            'Mg': 0.0,
            'Si': 1.0
        }],
        'elems': ['Al', 'Mg'],
        'expect': [1]
    }, {
        'bfs': [{
            'Al': 1.0,
            'Mg': 1.0,
            'Si': 0.0
        }, {
            'Al': 1.0,
            'Mg': 0.0,
            'Si': 1.0
        }],
        'elems': ['Al', 'Si'],
        'expect': [0]
    }, {
        'bfs': [{
            'Al': 1.0,
            'Mg': 1.0,
            'Si': 0.0,
            'X': 0.0
        }, {
            'Al': 1.0,
            'Mg': 0.0,
            'Si': 1.0,
            'X': 1.0
        }, {
            'Al': 0.0,
            'Mg': 1.0,
            'Si': 0.0,
            'X': 1.0
        }],
        'elems': ['Al', 'Si', 'X'],
        'expect': [0, 2]
    }]

    for test in tests:
        selection = rate_bf_subsets(test['elems'], test['bfs'])[0][1]
        assert selection == test['expect']


@pytest.mark.parametrize('test', [
    {
        'bfs': Polynomial(['Li', 'O', 'X', 'V']).get_basis_functions(),
        'basis_elems': [['Li', 'O'], ['X', 'V']]
    },
    {
        'bfs': Polynomial(['Li', 'O', 'V']).get_basis_functions(),
        'basis_elems': [['Li', 'O'], ['V', 'Li']]
    },
    {
        'bfs': Polynomial(['Li', 'O', 'V', 'X']).get_basis_functions(),
        'basis_elems': [['Li', 'O', 'X'], ['V', 'Li']]
    },
    {
        'bfs': Polynomial(['Li', 'O', 'V', 'X']).get_basis_functions(),
        'basis_elems': [['Li', 'O', 'X'], ['V', 'Li'], ['O', 'X']]
    },
    {
        'bfs': Polynomial(['Li', 'O', 'V', 'X']).get_basis_functions(),
        'basis_elems': [['Li', 'O', 'X'], ['V', 'Li'], ['O', 'X'], ['V', 'O', 'X']]
    },
    {
        'bfs': Polynomial(['Li', 'O', 'X']).get_basis_functions(),
        'basis_elems': [['Li', 'O', 'X'], ['O', 'Li']]
    },
])
def test_sublattice_bf_selection(test):

    selection = select_bf_subsets(test['basis_elems'], test['bfs'])

    # Confirm that all elements on each sublattice is distinguished
    bfs = test['bfs']
    for s, elems in zip(selection, test['basis_elems']):
        assert len(s) == len(elems) - 1
        distinguished = {}
        for bf_indx in s:
            for symb in product(elems, repeat=2):
                if symb[0] == symb[1]:
                    continue
                key = '-'.join(sorted(symb))
                diff = bfs[bf_indx][symb[0]] - bfs[bf_indx][symb[1]]

                disting = distinguished.get(key, False)
                distinguished[key] = disting or abs(diff) > 1e-4

        for v in distinguished.values():
            assert v, f'{distinguished}'


@pytest.mark.parametrize('test', [
    {
        'name1': 'c0',
        'name2': 'c1',
        'expect': True
    },
    {
        'name1': 'c1',
        'name2': 'c1',
        'expect': False
    },
    {
        'name1': 'c2_d0000_0',
        'name2': 'c1_0',
        'expect': False
    },
    {
        'name1': 'c0',
        'name2': 'c0',
        'expect': False
    },
    {
        'name1': 'c1_0',
        'name2': 'c1_1',
        'expect': True
    },
    {
        'name1': 'c4_d0000_10',
        'name2': 'c3_d9999_9',
        'expect': False
    },
    {
        'name1': 'c4_d0000_10',
        'name2': 'c4_d0000_9',
        'expect': False
    },
    {
        'name1': 'c2_d0200_9',
        'name2': 'c2_d0200_29',
        'expect': True
    },
])
def test_cname_lt(test):
    assert cname_lt(test['name1'], test['name2']) == test['expect']


@pytest.mark.parametrize('test', [{
    'bf': [{
        'Au': 1.0,
        'Cu': -1.0
    }],
    'cf': np.array([[1.0], [-1.0], [0.0]]),
    'expect': [{
        'Au': 1.0,
        'Cu': 0.0
    }, {
        'Au': 0.0,
        'Cu': 1.0
    }, {
        'Au': 0.5,
        'Cu': 0.5
    }]
}, {
    'bf': [{
        'Li': 1.0,
        'O': 0.0,
        'X': -1.0
    }, {
        'Li': 1.0,
        'O': -1.0,
        'X': 0.0
    }],
    'cf': np.array([[1.0, 1.0], [-0.5, -0.5]]),
    'expect': [{
        'Li': 1.0,
        'O': 0.0,
        'X': 0.0
    }, {
        'Li': 0.0,
        'O': 0.5,
        'X': 0.5
    }]
}])
def test_singlet2conc(test):
    conc = singlets2conc(test['bf'], test['cf'])
    assert len(conc) == len(test['expect'])
    for item1, item2 in zip(conc, test['expect']):
        assert item1 == item2


def test_aic():
    mse = 2.0
    n_feat = 3
    n_data = 5

    # expect = 2 * n_feat + n_data * np.log(mse)
    expect = 9.465735902799727
    assert expect == pytest.approx(aic(mse, n_feat, n_data))

    # Test with arrays and random data
    N = 20
    n_feat = np.random.choice(np.arange(1, 1000), size=N)
    n_data = np.random.choice(np.arange(1, 1000), size=N)
    mse = np.random.random(N) + 1e-6  # Add small constant to avoid 0

    calculated = aic(mse, n_feat, n_data)
    expect = 2 * n_feat + n_data * np.log(mse)
    assert np.allclose(calculated, expect)


def test_aicc():
    mse = 2.0
    n_feat = 3
    n_data = 5
    expect = 6.0 + 5 * np.log(mse) + 24.0
    assert expect == pytest.approx(aicc(mse, n_feat, n_data))


def test_bic():
    # Test with a pre-calculated example
    mse = 2.0
    n_feat = 3
    n_data = 5
    # expect = 3.0 * np.log(5) + 5 * np.log(mse)
    expect = 8.294049640102028
    assert expect == pytest.approx(bic(mse, n_feat, n_data))

    # Test with arrays and random data
    N = 20
    n_feat = np.random.choice(np.arange(1, 1000), size=N)
    n_data = np.random.choice(np.arange(1, 1000), size=N)
    mse = np.random.random(N) + 1e-6  # Add small constant to avoid 0

    calculated = bic(mse, n_feat, n_data)
    expect = np.log(n_data) * n_feat + n_data * np.log(mse)
    assert np.allclose(calculated, expect)


@pytest.mark.parametrize('fname,expect', [
    ('data.csv', '.csv'),
    ('file', ''),
    ('double_ext.csv.json', '.json'),
])
def test_get_file_extension(fname, expect):
    assert get_extension(fname) == expect


@pytest.mark.parametrize(
    'test',
    [
        {
            'fname': 'data.csv',
            'ext': '.csv',
            'expect': 'data.csv'
        },
        {
            'fname': 'data',
            'ext': '.json',
            'expect': 'data.json'
        },
        # Some cases where we expect to fail
        {
            'fname': 'data.json',
            'ext': '.csv',
            'expect': None
        },
        {
            'fname': 'data.json',
            'ext': '',
            'expect': None
        },
        {
            'fname': '',
            'ext': '.csv',
            'expect': None
        },
    ])
def test_add_proper_file_extension(test):
    expect = test['expect']
    if expect is None:
        # We should raise
        with pytest.raises(ValueError):
            add_file_extension('data.json', '.csv')
    else:
        fname = add_file_extension(test['fname'], test['ext'])
        assert fname == expect


def test_sort_cf_names():
    tests = [{
        'names': ['c1_8', 'c0_0', 'c1_1'],
        'expect': ['c0_0', 'c1_1', 'c1_8']
    }, {
        'names': [
            'c2_d0010_0_00',
            'c2_d0009_0_00',
            'c0_0',
            'c1_1',
        ],
        'expect': ['c0_0', 'c1_1', 'c2_d0009_0_00', 'c2_d0010_0_00']
    }, {
        'names': [
            'c3_d0008_0_00',
            'c2_d0009_0_00',
            'c0_0',
            'c1_1',
        ],
        'expect': ['c0_0', 'c1_1', 'c2_d0009_0_00', 'c3_d0008_0_00']
    }]

    for test in tests:
        sorted_names = sort_cf_names(test['names'])
        assert isinstance(sorted_names, list)
        assert sorted_names == test['expect']


def test_split_dataset():
    X = np.zeros((100, 10))
    y = np.zeros(100)

    # Case 1: Split without specifying groups
    partitions = split_dataset(X, y, nsplits=5)

    for p in partitions:
        assert p['train_X'].shape == (80, 10)
        assert len(p['train_y']) == 80
        assert p['validate_X'].shape == (20, 10)
        assert len(p['validate_y']) == 20

    # Case 2: Specify groups and check that entries that belonds
    # to the same groups is not split accross different partitions

    groups = []
    for i in range(len(X)):
        y[i] = i % 20
        X[i, :] = i % 20
        X = X.astype(int)
        y = y.astype(int)
        groups.append(i % 20)

    partitions = split_dataset(X, y, nsplits=5, groups=groups)
    for p in partitions:
        groups_train = set()
        groups_validate = set()

        flatX = p['train_X'].ravel().tolist()
        groups_train = groups_train.union(set(flatX))
        groups_train = groups_train.union(set(p['train_y']))

        flatX = p['validate_X'].ravel().tolist()
        groups_validate = groups_validate.union(set(flatX))
        groups_validate = groups_validate.union(set(p['validate_y']))

        # Make sure that the intersection between groups_train and groups
        # validate is an empty set
        assert not groups_train.intersection(groups_validate)


def test_common_cf_names(db_name):
    db = connect(db_name)
    table = 'polynomial_cf'
    atoms = bulk('Au')
    cfs = [{
        'c1_1': 0.0,
        'c2_d0000_0_00': 1.0,
        'c3_d0000_0_000': -1.0,
    }, {
        'c1_1': 0.0,
        'c2_d0000_0_00': 1.0,
        'c3_d0000_0_000': -1.0,
    }, {
        'c1_1': 0.2,
        'c3_d0000_0_000': 2.0
    }]
    for cf in cfs:
        db.write(atoms, external_tables={table: cf})

    ids = set([1, 2, 3])
    with connect(db_name) as db:
        cur = db.connection.cursor()
        common = common_cf_names(ids, cur, table)

    expect_common = set(['c1_1', 'c3_d0000_0_000'])
    assert common == expect_common


@pytest.mark.parametrize('test', [
    {
        'A_lb': np.array([[1.0, 1.0]]),
        'b_lb': np.array([0.0]),
        'A_eq': None,
        'b_eq': None,
        'c_lb': np.array([1.0, 1.0]),
        'd': -0.1,
        'expect': True
    },
    {
        'A_lb': np.array([[1.0, 1.0]]),
        'b_lb': np.array([0.0]),
        'A_eq': None,
        'b_eq': None,
        'c_lb': np.array([1.0, 1.0]),
        'd': 0.1,
        'expect': False
    },
    {
        'A_lb': np.array([[1.0, 1.0, 0.0]]),
        'b_lb': np.array([0.0]),
        'A_eq': np.array([[1.0, 0.0, -1.0]]),
        'b_eq': np.array([0.0]),
        'c_lb': np.array([1.0, 1.0, 0.0]),
        'd': 0.1,
        'expect': False
    },
])
def test_constraint_is_redundant(test):
    assert constraint_is_redundant(test['A_lb'], test['b_lb'], test['c_lb'], test['d'],
                                   test['A_eq'], test['b_eq']) == test['expect']


@pytest.mark.parametrize(
    'test',
    [
        # Paulraj et al.: Example 2.2
        {
            'A_lb':
                np.array([[-2.0, -1.0], [-4.0, 0.0], [-1.0, -3.0], [-1.0, -2.0], [0.0, -1.0],
                          [1.0, 1.0]]),
            'b_lb':
                np.array([-8.0, -15.0, -9.0, -14.0, -4.0, -5.0]),
            'A_lb_expect':
                np.array([[-2.0, -1.0], [-4.0, 0.0], [-1.0, -3.0]]),
            'b_lb_expect':
                np.array([-8.0, -15.0, -9.0])
        },
        # Telgen
        {
            'A_lb':
                np.array([[-1.0, 1.0], [-2.0, -1.0], [-1.0, 0.0], [1.0, -2.0], [0.0, -2.0],
                          [-1.0, -1.0]]),
            'b_lb':
                np.array([-2.0, -7.0, -2.0, -4.0, -5.0, -4.0]),
            'A_lb_expect':
                np.array([[-1.0, 0.0], [1.0, -2.0], [0.0, -2.0], [-1.0, -1.0]]),
            'b_lb_expect':
                np.array([-2.0, -4.0, -5.0, -4.0])
        },
    ],
    ids=['Paulraj', 'Telgen'])
def test_remove_redundant_constraints(test):
    # References:
    #
    # Paulraj et al.
    # Paulraj, S., and P. Sumathi. "A comparative study of redundant constraints
    # identification methods in linear programming problems." Mathematical Problems
    # in Engineering 2010 (2010).
    #
    # Telgen
    # Telgen, Jan. "Identifying redundant constraints and implicit equalities in
    # systems of linear constraints." Management Science 29.10 (1983): 1209-1222.
    A, b = remove_redundant_constraints(test['A_lb'], test['b_lb'])
    assert np.allclose(A, test['A_lb_expect'])
    assert np.allclose(b, test['b_lb_expect'])


@pytest.mark.parametrize('test', [{
    'A': np.array([[1.0, 1.0], [2.0, 2.0]]),
    'b': np.array([1.0, 2.0]),
    'A_expect': [[1.0, 1.0]],
    'b_expect': [1.0]
}, {
    'A': np.array([[1.0, 1.0, 0.3], [2.0, 2.0, 0.6]]),
    'b': np.array([1.0, 2.0, 0.0]),
    'A_expect': [[1.0, 1.0, 0.3]],
    'b_expect': [1.0]
}, {
    'A': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    'b': np.array([150.0, 300.0, 450.0]),
    'A_expect': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    'b_expect': [150.0, 300.0]
}, {
    'A': np.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [4.0, 3.0, 2.0]]),
    'b': np.array([50.0, 158.0, 142.0]),
    'A_expect': [[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]],
    'b_expect': [50.0, 158.0]
}])
def test_remove_redundant_equations(test):
    A, b = remove_redundant_equations(test['A'], test['b'])
    assert np.allclose(A, test['A_expect'])
    assert np.allclose(b, test['b_expect'])
