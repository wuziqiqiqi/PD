import pytest
from cleases.montecarlo.averager import Averager


def test_add_numbers():
    avg = Averager(ref_value=2.0)
    avg += 1.0
    avg += 2.0
    avg += 3.0
    assert avg.mean == pytest.approx(2.0)


def test_add_numbers_ref_zero():
    avg = Averager(ref_value=0.0)
    avg += 1.0
    avg += 2.0
    avg += 3.0
    assert avg.mean == pytest.approx(2.0)


def test_merge_two_averagers_same_ref():
    avg1 = Averager(ref_value=3.0)
    avg2 = Averager(ref_value=3.0)

    avg1 += 1.0
    avg1 += 2.0
    avg1 += 3.0
    avg2 += 4.0
    avg2 += 5.0
    avg2 += 6.0
    avg3 = avg1 + avg2
    assert avg3.mean == pytest.approx(3.5)


def test_merge_two_averagers_different_ref():
    avg1 = Averager(ref_value=3.0)
    avg2 = Averager(ref_value=-1.7)

    avg1 += 1.0
    avg1 += 2.0
    avg1 += 3.0
    avg2 += 4.0
    avg2 += 5.0
    avg2 += 6.0
    avg3 = avg1 + avg2
    assert avg3.mean == pytest.approx(3.5)
