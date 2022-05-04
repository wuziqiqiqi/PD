import pytest
import numpy as np
from clease.geometry import max_sphere_dia_in_cell


@pytest.mark.parametrize(
    "cell, expect",
    [
        # Ensure both "int" and "float" types are OK
        (np.diag([1, 1, 1]).astype(int), 1.0),
        (np.diag([1, 1, 1]).astype(float), 1.0),
        (np.array([[0.0, 1.9, 1.9], [1.9, 0.0, 1.9], [1.9, 1.9, 0.0]]), 2.1939310229205775),
    ],
)
def test_sphere_in_cell(cell, expect):
    dia = max_sphere_dia_in_cell(cell)
    assert pytest.approx(dia) == expect


def test_sphere_in_cube():
    """Test that a cubic cell where we rotate the y-vector
    around the z-axis recovers abs(cos(theta)).
    """
    ang = np.linspace(0.0, 2 * np.pi, 200, endpoint=True)

    def rot_z(t):
        """Helper function to make the rotation matrix around the z-axis
        at angle theta."""
        R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
        return R

    def make_cell(t):
        x = np.array([1, 0, 0])
        R = rot_z(t)
        y = R.dot([0, 1, 0])
        z = np.array([0, 0, 1])
        return np.vstack([x, y, z])

    y = [max_sphere_dia_in_cell(make_cell(a)) for a in ang]

    assert pytest.approx(y) == np.abs(np.cos(ang))
