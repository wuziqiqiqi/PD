import pytest
import numpy as np
from ase.build import bulk
from clease.geometry import max_sphere_dia_in_cell, supercell_which_contains_sphere


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


@pytest.mark.parametrize("dia", [15, 30, 35, 40, 41.5])
def test_sphere_in_sc(dia):
    # Use a cubic cell, easier to reason about how much expansion a sphere causes.
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=3.3, cubic=True)

    cell_vectors = np.linalg.norm(atoms.get_cell(), axis=1)
    assert (cell_vectors < dia).all()

    sc = supercell_which_contains_sphere(atoms, dia)

    cell_vectors = np.linalg.norm(sc.get_cell(), axis=1)
    assert (cell_vectors >= dia).all()
    # The cell vector shouldn't be overly large.
    assert (cell_vectors < 1.3 * dia).all()


@pytest.mark.parametrize(
    "atoms, expect",
    [
        (bulk("NaCl", crystalstructure="rocksalt", a=3.8, cubic=True), (11, 11, 11)),
        (bulk("NaCl", crystalstructure="rocksalt", a=3.8, cubic=False), (19, 19, 19)),
        (bulk("Au", crystalstructure="fcc", a=3.6, cubic=False), (20, 20, 20)),
    ],
)
def test_sphere_repeats(atoms, expect):
    dia = 40

    sc = supercell_which_contains_sphere(atoms, dia)
    assert "repeats" in sc.info
    assert (sc.info["repeats"] == expect).all()
