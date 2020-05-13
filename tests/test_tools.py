import unittest
from ase.build import bulk
from clease.tools import (
    min_distance_from_facet, factorize, all_integer_transform_matrices,
    species_chempot2eci, bf2matrix, rate_bf_subsets, select_bf_subsets,
    cname_lt, singlets2conc, aic, aicc, bic,
    get_extension, add_file_extension, equivalent_deco,
    sort_cf_names, split_dataset
)
from clease.basis_function import Polynomial
from itertools import product
import numpy as np


class TestTools(unittest.TestCase):
    def test_equivalent_deco(self):
        predict_result=[{'deco': [1,2,3,4],
                         'equiv_site': [[0,1,2]],
                         'result': [[1,2,3,4],
                                    [1,3,2,4],
                                    [2,1,3,4],
                                    [2,3,1,4],
                                    [3,1,2,4],
                                    [3,2,1,4]]},
                        {'deco': [1, 2, 3, 4],
                         'equiv_site': [[0, 3]],
                         'result': [[1, 2, 3, 4],
                                    [4, 2, 3, 1]],},
                        {'deco': [1, 2, 3, 4],
                         'equiv_site': [],
                         'result': [[1, 2, 3, 4]]}
                        ]
        method_result=[]
        for dict_list in predict_result:
            method_result.append(equivalent_deco(dict_list['deco'],dict_list['equiv_site']))

        for count,result_method in enumerate(method_result):
            self.assertListEqual(predict_result[count]['result'], result_method)


    def test_min_distance_from_facet(self):
        a = 4.0
        atoms = bulk('Al', crystalstructure='sc', a=a)

        x = [3.0, 3.5, 3.9]
        dist = min_distance_from_facet(x, atoms.get_cell())
        self.assertAlmostEqual(dist, 0.1)

    def test_factorize(self):
        fact = sorted(list(factorize(10)))
        self.assertEqual(fact, [2, 5])

        fact = sorted(list(factorize(16)))
        self.assertEqual(fact, [2, 2, 2, 2])

        fact = sorted(list(factorize(24)))
        self.assertEqual(fact, [2, 2, 2, 3])

    def test_all_int_matrices(self):
        _ = all_integer_transform_matrices(10)

    def test_species_chempot2eci(self):
        tests = [
            {
                'species': {'Au': 1.5},
                'bf_list': [{'Au': 1.0, 'Cu': -1.0}],
                'expect': {'c1_0': 1.5}
            },
            {
                'species': {'Au': 1.5, 'Cu': 0.5},
                'bf_list': [{'Au': 0.3, 'Cu': 1.2, 'X': 3.0},
                            {'Au': -0.3, 'Cu': 1.2, 'X': -3.0}],
                'expect': {'c1_0': 65/24, 'c1_1': -55/24}
            }
        ]

        for i, test in enumerate(tests):
            eci = species_chempot2eci(test['bf_list'], test['species'])
            msg = 'Test #{} failed '.format(i)
            msg += 'Setup: {}'.format(test)
            for k, v in eci.items():
                self.assertAlmostEqual(v, test['expect'][k], msg=msg)

    def test_bf2matrix(self):
        tests = [
            {
                'bf': [{'Al': 1.0, 'Mg': -1.0}],
                'expect': np.array([[1.0, -1.0]])
            },
            {
                'bf': [{'Li': 1.0, 'O': 0.0, 'X': 0.0},
                       {'Li': 0.0, 'O': 1.0, 'X': 0.0}],
                'expect': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            }
        ]

        for i, test in enumerate(tests):
            mat = bf2matrix(test['bf'])
            self.assertTrue(np.allclose(mat, test['expect']))

    def test_subset_selection(self):
        tests = [
            {
                'bfs': [{'Al': 1.0, 'Mg': 1.0, 'Si': 0.0},
                        {'Al': 1.0, 'Mg': 0.0, 'Si': 1.0}],
                'elems': ['Al', 'Mg'],
                'expect': [1]
            },
            {
                'bfs': [{'Al': 1.0, 'Mg': 1.0, 'Si': 0.0},
                        {'Al': 1.0, 'Mg': 0.0, 'Si': 1.0}],
                'elems': ['Al', 'Si'],
                'expect': [0]
            },
            {
                'bfs': [{'Al': 1.0, 'Mg': 1.0, 'Si': 0.0, 'X': 0.0},
                        {'Al': 1.0, 'Mg': 0.0, 'Si': 1.0, 'X': 1.0},
                        {'Al': 0.0, 'Mg': 1.0, 'Si': 0.0, 'X': 1.0}],
                'elems': ['Al', 'Si', 'X'],
                'expect': [0, 2]
            }
        ]

        for i, test in enumerate(tests):
            selection = rate_bf_subsets(test['elems'], test['bfs'])[0][1]
            self.assertEqual(selection, test['expect'])

    def test_sublattice_bf_selection(self):
        tests = [
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
                'basis_elems': [['Li', 'O', 'X'], ['V', 'Li'], ['O', 'X'],
                                ['V', 'O', 'X']]
            },
            {
                'bfs': Polynomial(['Li', 'O', 'X']).get_basis_functions(),
                'basis_elems': [['Li', 'O', 'X'], ['O', 'Li']]
            },
        ]

        for i, test in enumerate(tests):
            selection = select_bf_subsets(test['basis_elems'], test['bfs'])

            # Confirm that all elements on each sublattice is distinguished
            bfs = test['bfs']
            for s, elems in zip(selection, test['basis_elems']):
                self.assertEqual(len(s), len(elems)-1)
                distinguished = {}
                for bf_indx in s:
                    for symb in product(elems, repeat=2):
                        if symb[0] == symb[1]:
                            continue
                        key = '-'.join(sorted(symb))
                        diff = bfs[bf_indx][symb[0]] - bfs[bf_indx][symb[1]]

                        disting = distinguished.get(key, False)
                        distinguished[key] = disting or abs(diff) > 1e-4

                for k, v in distinguished.items():
                    self.assertTrue(v, msg='{}'.format(distinguished))

    def test_cname_lt(self):
        tests = [
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
            ]

        for t in tests:
            self.assertEqual(cname_lt(t['name1'], t['name2']), t['expect'])

    def test_singlet2conc(self):
        tests = [
            {
                'bf': [{'Au': 1.0, 'Cu': -1.0}],
                'cf': np.array([[1.0], [-1.0], [0.0]]),
                'expect': [{'Au': 1.0, 'Cu': 0.0},
                           {'Au': 0.0, 'Cu': 1.0},
                           {'Au': 0.5, 'Cu': 0.5}]
            },
            {
                'bf': [{'Li': 1.0, 'O': 0.0, 'X': -1.0},
                       {'Li': 1.0, 'O': -1.0, 'X': 0.0}],
                'cf': np.array([[1.0, 1.0], [-0.5, -0.5]]),
                'expect': [{'Li': 1.0, 'O': 0.0, 'X': 0.0},
                           {'Li': 0.0, 'O': 0.5, 'X': 0.5}]
            }]

        for t in tests:
            conc = singlets2conc(t['bf'], t['cf'])
            self.assertEqual(len(conc), len(t['expect']))
            for item1, item2 in zip(conc, t['expect']):
                self.assertDictEqual(item1, item2)

    def test_aic(self):
        mse = 2.0
        n_feat = 3
        n_data = 5
        expect = 6.0 + 5*np.log(mse)
        self.assertAlmostEqual(expect, aic(mse, n_feat, n_data))

    def test_aicc(self):
        mse = 2.0
        n_feat = 3
        n_data = 5
        expect = 6.0 + 5*np.log(mse) + 24.0
        self.assertAlmostEqual(expect, aicc(mse, n_feat, n_data))

    def test_bic(self):
        mse = 2.0
        n_feat = 3
        n_data = 5
        expect = 3.0*np.log(5) + 5*np.log(mse)
        self.assertAlmostEqual(expect, bic(mse, n_feat, n_data))

    def test_get_filename(self):
        tests = [
            {
                'fname': 'data.csv',
                'expect': 'csv'
            },
            {
                'fname': 'file',
                'expect': ''
            },
            {
                'fname': 'double_ext.csv.json',
                'expect': 'json'
            }
        ]
        for t in tests:
            ext = get_extension(t['fname'])
            self.assertEqual(ext, t['expect'])

    def test_add_proper_file_extension(self):
        tests = [
            {
                'fname': 'data.csv',
                'ext': 'csv',
                'expect': 'data.csv',
            },
            {
                'fname': 'data',
                'ext': 'json',
                'expect': 'data.json'
            }
        ]

        for t in tests:
            fname = add_file_extension(t['fname'], t['ext'])
            self.assertEqual(fname, t['expect'])

        with self.assertRaises(ValueError):
            add_file_extension('data.json', 'csv')

    def test_sort_cf_names(self):
        tests = [
            {
                'names': ['c1_8', 'c0_0', 'c1_1'],
                'expect': ['c0_0', 'c1_1', 'c1_8']
            },
            {
                'names': ['c2_d0010_0_00', 'c2_d0009_0_00', 'c0_0', 'c1_1', ],
                'expect': ['c0_0', 'c1_1', 'c2_d0009_0_00', 'c2_d0010_0_00']
            },
            {
                'names': ['c3_d0008_0_00', 'c2_d0009_0_00', 'c0_0', 'c1_1', ],
                'expect': ['c0_0', 'c1_1', 'c2_d0009_0_00', 'c3_d0008_0_00']
            }
        ]

        for test in tests:
            sorted_names = sort_cf_names(test['names'])
            self.assertListEqual(sorted_names, test['expect'])

    def test_split_dataset(self):
        X = np.zeros((100, 10))
        y = np.zeros(100)

        # Case 1: Split without specifying groups
        partitions = split_dataset(X, y, nsplits=5)

        for p in partitions:
            self.assertEqual(p['train_X'].shape, (80, 10))
            self.assertEqual(len(p['train_y']), 80)
            self.assertEqual(p['validate_X'].shape, (20, 10))
            self.assertEqual(len(p['validate_y']), 20)

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
            self.assertEqual(set(), groups_train.intersection(groups_validate))





if __name__ == '__main__':
    unittest.main()
