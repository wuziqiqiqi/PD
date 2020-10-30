from ase.build import bulk
import numpy as np
from clease.montecarlo import RandomFlip, RandomSwap


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
