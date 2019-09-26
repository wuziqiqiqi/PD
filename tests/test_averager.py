import unittest
from clease.montecarlo.averager import Averager


class TestAverager(unittest.TestCase):
    def test_add_numbers(self):
        avg = Averager(ref_value=2.0)
        avg += 1.0
        avg += 2.0
        avg += 3.0
        self.assertAlmostEqual(2.0, avg.mean)

    def test_add_numbers_ref_zero(self):
        avg = Averager(ref_value=0.0)
        avg += 1.0
        avg += 2.0
        avg += 3.0
        self.assertAlmostEqual(2.0, avg.mean)

    def test_merge_two_averagers_same_ref(self):
        avg1 = Averager(ref_value=3.0)
        avg2 = Averager(ref_value=3.0)

        avg1 += 1.0
        avg1 += 2.0
        avg1 += 3.0
        avg2 += 4.0
        avg2 += 5.0
        avg2 += 6.0
        avg3 = avg1 + avg2
        self.assertAlmostEqual(avg3.mean, 3.5)

    def test_merge_two_averagers_different_ref(self):
        avg1 = Averager(ref_value=3.0)
        avg2 = Averager(ref_value=-1.7)

        avg1 += 1.0
        avg1 += 2.0
        avg1 += 3.0
        avg2 += 4.0
        avg2 += 5.0
        avg2 += 6.0
        avg3 = avg1 + avg2
        self.assertAlmostEqual(avg3.mean, 3.5)

if __name__ == '__main__':
    unittest.main()