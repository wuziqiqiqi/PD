import pytest
from ase.build import bulk
import numpy as np
from clease.montecarlo import RandomFlip, RandomSwap, MixedSwapFlip, TooFewElementsError


def test_random_flip():
    np.random.seed(42)
    atoms = bulk('Au') * (4, 4, 4)
    symbs = ['Au', 'Cu', 'X']

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
    np.random.seed(42)
    atoms = bulk('Au') * (4, 4, 4)

    # Check that error is raised
    with pytest.raises(TooFewElementsError):
        swapper = RandomSwap(atoms)

    atoms.symbols[:4] = 'Cu'
    atoms.symbols[4:10] = 'X'
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
    np.random.seed(42)
    atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.0) * (4, 4, 4)

    # Cl sublattice should have fixed conc, Na sublattice should have
    # fixed chemical potential
    flip_idx = [atom.index for atom in atoms if atom.symbol == 'Na']
    swap_idx = [atom.index for atom in atoms if atom.symbol == 'Cl']

    # Insert some oxygen on the Cl sublattice
    atoms.symbols[swap_idx[:6]] = 'O'

    # Insert some vacancies on the Na sublattice
    atoms.symbols[flip_idx[:6]] = 'X'

    generator = MixedSwapFlip(atoms, swap_idx, flip_idx, ['Na', 'X'])

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
            assert old1 in ['Cl', 'O']
            assert old2 in ['Cl', 'O']
            assert name1 == name2
            assert generator.swapper.name_matches(move[0])
        elif len(move) == 1:
            assert move[0].index in flip_idx
            assert move[0].old_symb != move[0].new_symb
            assert move[0].old_symb in ['Na', 'X']
            assert move[0].new_symb in ['Na', 'X']
            assert generator.flipper.name_matches(move[0])
        else:
            raise RuntimeError("Generator produced move that is not swap and not a flip")
