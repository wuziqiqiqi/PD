import numpy as np
from cleases.montecarlo.observers.buffered_array import BufferedArray


def test_buffered_array():
    array = BufferedArray(size=3)
    array.push(1.0)
    array.push(2.0)
    array.push(3.0)

    assert np.allclose(array._buffer, [1.0, 2.0, 3.0])

    array.push(4.0)
    assert np.allclose(array._buffer, [4.0, 0.0, 0.0])


def load_buffer(buffer_file):
    with open(buffer_file, "r") as f:
        return [float(line.strip()) for line in f]


def test_buffered_array_file(buffer_file):
    array = BufferedArray(size=3, fname=buffer_file)
    for i in range(4):
        array.push(i)

    # Now the file should contain the first 3 records
    assert np.allclose(load_buffer(buffer_file), [0.0, 1.0, 2.0])

    for i in range(4, 7):
        array.push(i)

    # Now the file should contain the 6 first records
    assert np.allclose(load_buffer(buffer_file), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
