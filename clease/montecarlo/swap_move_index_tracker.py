from typing import Sequence, List, Dict
import random
import numpy as np
import ase


class SwapMoveIndexTracker:
    """
    Tracker for possible swap move indices.

    :param atoms: The ase Atoms object to be tracked.
    :param indices: List with indices that can be chosen from. If None, all indices
        can be chosen.
    """

    def __init__(self, atoms, indices: Sequence[int] = None):
        self.tracker: Dict[str, List[int]] = {}
        self.index_loc = None
        self._last_move = []

        self._init_tracker(atoms, indices=indices)

        # Cache of the unique symbols for faster lookup
        self.unique_symbols = self.get_unique_symbols()

    @staticmethod
    def _unique_symbols_from_atoms(atoms) -> List[str]:
        return sorted(set(atoms.symbols))

    def get_unique_symbols(self) -> List[str]:
        return list(self.tracker.keys())

    @property
    def num_symbols(self) -> int:
        return len(self.tracker)

    def __repr__(self):
        str_repr = f"SwapMoveIndexTracker at {hex(id(self))}\n"
        str_repr += f"Symbols tracked: {self.get_unique_symbols()}\n"
        str_repr += f"Tracker info: {self.tracker}\n"
        return str_repr

    def _init_tracker(self, atoms: ase.Atoms, indices: Sequence[int] = None) -> None:
        """Initialize the tracker with the numbers."""
        # Cache the unique symbols for faster access
        symbols = self._unique_symbols_from_atoms(atoms)

        # Track indices of all symbols
        self.tracker = {symb: [] for symb in symbols}

        # Track the location in self.tracker of each index
        self.index_loc = np.zeros(len(atoms), dtype=int)
        for atom in atoms:
            self.tracker[atom.symbol].append(atom.index)
            self.index_loc[atom.index] = len(self.tracker[atom.symbol]) - 1

        if indices is not None:
            self._filter_indices(indices)

    def _filter_indices(self, include: Sequence[int]) -> None:
        include_set = set(include)
        for s, idx in self.tracker.items():
            self.tracker[s] = list(set(idx).intersection(include_set))
            for i, num in enumerate(self.tracker[s]):
                self.index_loc[num] = i

        # Remove empty items
        self.tracker = {k: v for k, v in self.tracker.items() if v}

    def move_already_updated(self, system_changes) -> bool:
        """Return True if system_changes have already been taken into account.

        Parameters:

        system_changes:
        """
        return system_changes == self._last_move

    def update_swap_move(self, system_changes) -> None:
        """Update the atoms tracker.

        Parameters:

        system_changes:
        """
        if self.move_already_updated(system_changes):
            # This change has already been updated!
            return
        self._last_move = system_changes
        indx1 = system_changes[0].index
        indx2 = system_changes[1].index
        symb1 = system_changes[0].old_symb
        symb2 = system_changes[0].new_symb

        # Find the locations of the indices
        loc1 = self.index_loc[indx1]
        loc2 = self.index_loc[indx2]

        # Update the tracker and the locations
        self.tracker[symb1][loc1] = indx2
        self.index_loc[indx2] = loc1

        self.tracker[symb2][loc2] = indx1
        self.index_loc[indx1] = loc2

    def undo_last_swap_move(self) -> None:
        """Undo last swap move."""
        if not self._last_move:
            return
        opposite_change = []
        for change in self._last_move:
            opposite_change.append((change[0], change[2], change[1]))

        self.update_swap_move(opposite_change)
        self._last_move = []

    def get_random_indx_of_symbol(self, symbol: str) -> int:
        return random.choice(self.tracker[symbol])
