from copy import deepcopy
from collections import Counter
import pytest
from ase.build import bulk
import numpy as np
from scipy import stats
from clease.montecarlo import (
    RandomFlip,
    RandomSwap,
    MixedSwapFlip,
    TooFewElementsError,
    RandomFlipWithinBasis,
)


def test_random_flip():
    atoms = bulk("Au") * (4, 4, 4)
    symbs = ["Au", "Cu", "X"]

    all_indices = [None, [0, 4, 7, 32, 40]]
    for indices in all_indices:
        flipper = RandomFlip(symbs, atoms, indices=indices)
        allowed_indices = indices
        if allowed_indices is None:
            allowed_indices = list(range(len(atoms)))

        # Run 10 flips. Ensure that the move is valid
        for _ in range(10):
            move = flipper.get_single_trial_move()
            assert len(move) == 1
            assert move[0].old_symb in symbs
            assert move[0].new_symb in symbs
            assert move[0].old_symb != move[0].new_symb
            assert move[0].index in allowed_indices


def test_random_swap():
    atoms = bulk("Au") * (4, 4, 4)

    # Check that error is raised
    with pytest.raises(TooFewElementsError):
        swapper = RandomSwap(atoms)

    atoms.symbols[:4] = "Cu"
    atoms.symbols[4:10] = "X"
    unique_symbs = set(atoms.symbols)

    all_indices = [None, [0, 4, 7, 32, 40]]
    for indices in all_indices:
        swapper = RandomSwap(atoms, indices=indices)
        allowed_indices = indices
        if allowed_indices is None:
            allowed_indices = list(range(len(atoms)))

        # Run 10 flips. Ensure that the move is valid
        for _ in range(10):
            moves = swapper.get_single_trial_move()
            assert len(moves) == 2
            for move in moves:
                assert move.old_symb in unique_symbs
                assert move.new_symb in unique_symbs
                assert move.old_symb != move.new_symb
                assert move.index in allowed_indices

            # Confirm that the two moves actually constitute a swap
            assert moves[0].old_symb == moves[1].new_symb
            assert moves[0].new_symb == moves[1].old_symb
            assert moves[0].index != moves[1].index


def test_mixed_ensemble():
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=4.0) * (4, 4, 4)

    # Cl sublattice should have fixed conc, Na sublattice should have
    # fixed chemical potential
    flip_idx = [atom.index for atom in atoms if atom.symbol == "Na"]
    swap_idx = [atom.index for atom in atoms if atom.symbol == "Cl"]

    # Insert some oxygen on the Cl sublattice
    atoms.symbols[swap_idx[:6]] = "O"

    # Insert some vacancies on the Na sublattice
    atoms.symbols[flip_idx[:6]] = "X"

    generator = MixedSwapFlip(atoms, swap_idx, flip_idx, ["Na", "X"])

    num_moves = 100
    for _ in range(num_moves):
        move = generator.get_trial_move()

        # If the length of move is 2: Cl sublattice
        if len(move) == 2:
            idx1, old1, new1, name1 = move[0]
            idx2, old2, new2, name2 = move[1]
            assert idx1 != idx2
            assert idx1 in swap_idx
            assert idx2 in swap_idx
            assert old1 == new2
            assert old2 == new1
            assert old1 in ["Cl", "O"]
            assert old2 in ["Cl", "O"]
            assert name1 == name2
            assert generator.swapper.name_matches(move[0])
        elif len(move) == 1:
            assert move[0].index in flip_idx
            assert move[0].old_symb != move[0].new_symb
            assert move[0].old_symb in ["Na", "X"]
            assert move[0].new_symb in ["Na", "X"]
            assert generator.flipper.name_matches(move[0])
        else:
            raise RuntimeError("Generator produced move that is not swap and not a flip")


def test_random_flip_within_basis():
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=4.0) * (3, 3, 3)
    basis1 = [a.index for a in atoms if a.symbol == "Na"]
    basis2 = [a.index for a in atoms if a.symbol == "Cl"]

    basis = [basis1, basis2]
    symbs = [["Na", "X"], ["Cl", "O"]]

    # Check wrong length of symbols
    with pytest.raises(ValueError):
        RandomFlipWithinBasis([["Na", "X"]], atoms, basis)

    # Check non-unique symbols
    with pytest.raises(ValueError):
        RandomFlipWithinBasis([["Na", "X", "Na"], ["Na", "X"]], atoms, basis)

    # Check for index in two basis
    with pytest.raises(ValueError):
        basis2_cpy = deepcopy(basis2)
        basis2_cpy[2] = basis1[0]
        RandomFlipWithinBasis(symbs, atoms, [basis1, basis2_cpy])

    # Check index two times in the same basis
    with pytest.raises(ValueError):
        basis2_cpy = deepcopy(basis2)
        basis2_cpy[2] = basis2_cpy[0]
        RandomFlipWithinBasis(symbs, atoms, [basis1, basis2_cpy])

    flipper = RandomFlipWithinBasis(symbs, atoms, basis)
    num = 1000
    basis_count = [0, 0]

    # Test we get exactly half of the atoms in a flipper
    assert all(len(f.indices) == len(atoms) // 2 for f in flipper._flippers)
    assert all(len(f.symbols) == 2 for f in flipper._flippers)

    for _ in range(num):
        change = flipper.get_single_trial_move()[0]
        chosen_basis = 0 if change.index in basis1 else 1
        assert change.index in basis[chosen_basis]
        basis_count[chosen_basis] += 1
        assert change.new_symb in symbs[chosen_basis]
        assert change.old_symb in symbs[chosen_basis]

    # Check that the p-value for the null hypothesis (prob of selecting a basis is 0.5) is
    # larger than 0.05, i.e. we do not reject the null hypothesis.
    # if p < 0.05, it would be unlikely the basis count was chosen with equal probability.
    pval = stats.binom_test(basis_count, p=0.5)
    assert pval > 0.05, basis_count


@pytest.mark.parametrize("flip_prob", [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_mixed_swap_probability(flip_prob):
    """Test the flip_prob properly affects the ratio of flip moves and swap moves"""

    atoms = bulk("NaCl", crystalstructure="rocksalt", a=4.0, cubic=True) * (4, 4, 4)
    # Let all atoms be allowed to flip and swap. We only care about the distribution
    # of flips and swaps.
    N = len(atoms)
    generator = MixedSwapFlip(atoms, range(N), range(N), ["Na", "Cl"], flip_prob=flip_prob)

    def get_move_type():
        """Helper function, get a random move, and return the type of move"""
        changes = generator.get_single_trial_move()
        return changes[0].name

    num = 10_000
    c = Counter(get_move_type() for _ in range(num))

    # We can have 1 move type if flip_prob = 0 or 1
    # otherwise we should have 2
    assert 1 <= len(c) <= 2
    # A "flip_move" is the "success" case in the binom_test, it needs to be first
    counts = [c["flip_move"], c["swap_move"]]

    # See comment in test_random_flip_within_basis for discussion on p-values
    pval = stats.binom_test(counts, p=flip_prob)
    assert pval > 0.05
