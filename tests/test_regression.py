from clease import LinearRegression, Lasso, Tikhonov
import numpy as np
import unittest


class TestRegression(unittest.TestCase):
    def test_non_singular(self):
        x = np.linspace(0.0, 1.0, 20)

        y = 1.0 + 2.0*x - 4.0*x**2

        X = np.zeros((len(x), 3))
        X[:, 0] = 1.0
        X[:, 1] = x
        X[:, 2] = x**2

        linreg = LinearRegression()
        coeff = linreg.fit(X, y)

        # Test that fit works
        self.assertTrue(np.allclose(coeff, [1.0, 2.0, -4.0]))

        # Test that precision matrix gives correct result
        # in the case where it is not singular
        prec = np.linalg.inv(X.T.dot(X))
        prec_regr = linreg.precision_matrix(X)
        self.assertTrue(np.allclose(prec, prec_regr))

    def test_trivial_singular(self):
        x = np.linspace(0.0, 1.0, 20)

        y = 1.0 + 2.0*x - 4.0*x**2

        X = np.zeros((len(x), 4))
        X[:, 0] = 1.0
        X[:, 1] = x
        X[:, 2] = x**2
        X[:, 3] = x**2

        linreg = LinearRegression()
        coeff = linreg.fit(X, y)

        self.assertTrue(np.allclose(X.dot(coeff), y))
        linreg.precision_matrix(X)

    def test_complicated_singular(self):
        x = np.linspace(0.0, 1.0, 20)

        y = 1.0 + 2.0*x - 4.0*x**2

        X = np.zeros((len(x), 5))
        X[:, 0] = 1.0
        X[:, 1] = x
        X[:, 2] = x**2
        X[:, 3] = 0.1 - 0.2*x + 0.8*x**2
        X[:, 4] = -0.2 + 0.8*x

        linreg = LinearRegression()
        coeff = linreg.fit(X, y)

        self.assertTrue(np.allclose(X.dot(coeff), y))
        linreg.precision_matrix(X)


class Test_Tikhonov(unittest.TestCase):
    def test_get_instance_array(self):
        Test_dict = [{'alpha_min': 0.2,
                      'alpha_max': 0.5,
                      'num_alpha': 5,
                      'scale': 'log'},
                     {'alpha_min': 1,
                      'alpha_max': 5,
                      'num_alpha': 10,
                      'scale': 'log'},
                     {'alpha_min': 2,
                      'alpha_max': 4,
                      'num_alpha': 4,
                      'scale': 'etc'}]
        True_alpha = []
        Tikreg = Tikhonov()
        predict_alpha = []
        for test in Test_dict:
            if test['scale'] == 'log':
                True_alpha.append(np.ndarray.tolist(np.logspace(np.log10(test['alpha_min']), np.log10(test['alpha_max']),
                                                                int(test['num_alpha']), endpoint=True)))
            else:
                True_alpha.append(np.ndarray.tolist(np.linspace(test['alpha_min'], test['alpha_max'], int(test['num_alpha']),
                                                                endpoint=True)))

            Tik_list = Tikreg.get_instance_array(test['alpha_min'], test['alpha_max'], test['num_alpha'], test['scale'])
            predict_alpha.append([i.alpha for i in Tik_list])

        for true, predict in zip(True_alpha, predict_alpha):
            self.assertListEqual(true, predict)

    def test_isscalar(self):
        Test_dict=[{'alpha': 10.0,
                    'bool': True},
                   {'alpha': 5.0,
                    'bool': True},
                   {'alpha': [20, 20, 10],
                    'bool': False},
                   {'alpha': 10,
                    'bool': False}]
        Tik = Tikhonov()
        predict_list = []

        for loop in Test_dict:
            Tik.alpha = loop['alpha']
            predict_list.append(Tik.is_scalar())

        for test, predict in zip(Test_dict, predict_list):
            if predict:
                Tik.alpha = test['alpha']
                self.assertEqual(Tik.get_scalar_parameter(), test['alpha'])


class Test_Lasso(unittest.TestCase):
    def test_fit(self):
        input_set=[{'X': [[1, 2], [3, 4], [5, 3]],
                    'Y': [4, 5, 2]},
                   {'X': [[3, 1, 5], [2, 4, 4], [6, 5, 3]],
                    'Y': [4, 5, 2]},
                   {'X': [[1, 2, 6, 1], [3, 4, 3, 8], [5, 3, 4, 6]],
                    'Y': [9, 1, 2]},
                   {'X': [[1, 2, 12, 43, 2], [3, 4, 3, 12, 21], [5, 3, 21, 17, 20]],
                    'Y': [4, 5, 2]},
                   {'X': [[1, 2, 6, 1, 0, 12], [3, 4, 6, 7, 8, 12], [5, 3, 3, 4, 5, 6]],
                    'Y': [10, 31, 12]}
                   ]
        import sklearn.linear_model
        sk_lasso = sklearn.linear_model.Lasso(alpha=1, fit_intercept=False, copy_X=True, normalize=True, max_iter=1e6)
        cl_lasso = Lasso(alpha=1)

        true_coeff=[]
        predict_coeff=[]
        for data in input_set:
            sk_lasso.fit(data['X'], data['Y'])
            true_coeff.append(np.ndarray.tolist(sk_lasso.coef_))
            predict_coeff.append(np.ndarray.tolist(cl_lasso.fit(data['X'], data['Y'])))

        for true, preict in zip(true_coeff, predict_coeff):
            self.assertListEqual(true, preict)

    def test_get_instance_array(self):
        Test_dict=[{'alpha_min': 0.2,
                    'alpha_max': 0.5,
                    'num_alpha': 5,
                   'scale': 'log'},
                   {'alpha_min': 1,
                    'alpha_max': 5,
                    'num_alpha': 10,
                    'scale': 'log'},
                   {'alpha_min': 2,
                    'alpha_max': 4,
                    'num_alpha': 4,
                    'scale': 'etc'}]
        True_alpha = []
        lasreg = Lasso()
        predict_alpha = []
        for test in Test_dict:
            if test['scale'] == 'log':
                True_alpha.append(np.ndarray.tolist(np.logspace(np.log10(test['alpha_min']), np.log10(test['alpha_max']),
                                    int(test['num_alpha']), endpoint=True)))
            else:
                True_alpha.append(np.ndarray.tolist(np.linspace(test['alpha_min'], test['alpha_max'], int(test['num_alpha']),
                                    endpoint=True)))

            las_list = lasreg.get_instance_array(test['alpha_min'], test['alpha_max'],test['num_alpha'],test['scale'])
            predict_alpha.append([i.alpha for i in las_list])

        for true, predict in zip(True_alpha, predict_alpha):
            self.assertListEqual(true,predict)

    def test_get_scalar_parameter(self):
        Test_dict=[{'alpha': 10.0,
                   'bool': True},
                   {'alpha': 5.0,
                    'bool': True},
                   {'alpha': [20, 20, 10],
                    'bool': False},
                   {'alpha': 10,
                    'bool': False}]
        las = Lasso()
        for test in Test_dict:
            if las.is_scalar():
                las.alpha = test['alpha']
                self.assertEqual(las.get_scalar_parameter(), test['alpha'])

if __name__ == '__main__':
    unittest.main()
