import unittest
import numpy as np
from clease.gramSchmidthMonomials import GramSchmidtMonimial


def bf_list_equal(d1, d2):
    for i1, i2 in zip(d1, d2):
        for k in i1.keys():
            if abs(i1[k] - i2[k]) > 1e-10:
                return False
    return True


class TestGramSchmidtMonimial(unittest.TestCase):
    def test_spin_dicts(self):
        tests = [
            {"symbols": ["Au", "Cu"], "expect": [{"Au": 1.0, "Cu": -1.0}]},
            {
                "symbols": ["Au", "Cu", "X"],
                "expect": [
                    {"Au": np.sqrt(3.0 / 2.0), "Cu": -np.sqrt(3.0 / 2.0), "X": 0.0},
                    {
                        "Au": 1.0 / np.sqrt(2),
                        "Cu": 1.0 / np.sqrt(2.0),
                        "X": -np.sqrt(2.0),
                    },
                ],
            },
        ]

        for test in tests:
            gram_schmidt = GramSchmidtMonimial(len(test["symbols"]))
            gram_schmidt.build()
            sp_dict = gram_schmidt.basis_functions(test["symbols"])

            msg = "Expected\n{}\nGot\n{}".format(test["expect"], sp_dict)
            self.assertTrue(bf_list_equal(test["expect"], sp_dict), msg=msg)

    def test_orthonormality(self):
        for num_symbs in range(2, 10):
            gram_schmidt = GramSchmidtMonimial(num_symbs)
            gram_schmidt.build()
            bf_array = np.array(gram_schmidt.bf_values)
            identity = np.eye(num_symbs)
            dotProd = bf_array.T.dot(bf_array) / num_symbs
            self.assertTrue(np.allclose(dotProd, identity, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
