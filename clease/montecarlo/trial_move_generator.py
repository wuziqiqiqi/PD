from typing import Sequence, List, Set
from abc import abstractmethod, ABC
from ase import Atoms
import numpy as np
from clease.montecarlo.constraints import MCConstraint
from clease.tools import SystemChange
from clease.montecarlo.swap_move_index_tracker import SwapMoveIndexTracker

__all__ = ('TrialMoveGenerator', 'RandomFlip', 'RandomSwap')

DEFAULT_MAX_ATTEMPTS = 10000


class TrialMoveGenerator(ABC):
    """
    Class for producing trial moves.

    :param max_attempts: Maximum number of attempts to try to find a move that passes
        the constraints. If not constraints are added, this has no effect.
    """

    def __init__(self, max_attempts: int = DEFAULT_MAX_ATTEMPTS):
        super().__init__()
        self._constraints = []
        self.max_attempts = max_attempts

    def initialize(self, atoms: Atoms) -> None:
        """
        Initialize the generator.

        :param atoms: Atoms object used in the simulation
        """

    def remove_constraints(self) -> None:
        """
        Remove all constraints
        """
        self._constraints = []

    @property
    def name(self):
        return type(self).__name__

    def add_constraint(self, cnst: MCConstraint):
        """
        Add a constraint to the generator

        :param cnst: Constraint that must be satisfied for all trial moves
        """
        self._constraints.append(cnst)

    def _change_ok(self, system_changes: Sequence[SystemChange]) -> bool:
        for cnst in self._constraints:
            if not cnst(system_changes):
                return False
        return True

    @abstractmethod
    def get_single_trial_move(self) -> Sequence[SystemChange]:
        """
        Return a single trial move, must be implemented in sub-classes
        """

    def get_trial_move(self) -> Sequence[SystemChange]:
        """
        Produce a trial move that is consistent with all cosntraints
        """
        for _ in range(self.max_attempts):
            move = self.get_single_trial_move()
            if self._change_ok(move):
                return move
        raise RuntimeError("Can not produce a valid trial move")

    def on_move_accepted(self, change: SystemChange) -> None:
        """
        Callback that is called by Monte Carlo after each accepted move

        :param change: Trial move
        """

    def on_move_rejected(self, change: SystemChange) -> None:
        """
        Callback that is called after a move is rejected

        :param change: Trial move
        """


class RandomFlip(TrialMoveGenerator):
    """
    Generate trial moves where the symbol at a given site is flipped

    :param symbols: Set with all symbols considered in a move
    :param atoms: Atoms object for the simulation
    :param indices: List with all indices that should be considered. If None, all indices are
        considered
    """

    def __init__(self, symbols: Set[str], atoms: Atoms, indices: List[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.symbols = symbols
        self.atoms = atoms

        if indices is None:
            self.indices = list(range(len(self.atoms)))
        else:
            self.indices = indices

    def get_single_trial_move(self) -> List[SystemChange]:
        pos = np.random.choice(self.indices)
        old_symb = self.atoms[pos].symbol
        new_symb = np.random.choice([s for s in self.symbols if s != old_symb])
        return [SystemChange(index=pos, old_symb=old_symb, new_symb=new_symb)]


class RandomSwap(TrialMoveGenerator):
    """
    Produce random swaps

    :param atoms: Atoms object in the MC simulation
    :param indices: List with indices that can be chosen from. If None, all indices
        can be chosen.
    """

    def __init__(self, atoms: Atoms, indices: List[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.tracker = SwapMoveIndexTracker()
        self.indices = indices
        self.initialize(atoms)

    def initialize(self, atoms: Atoms) -> None:
        """
        Initializes the atoms tracker
        """
        self.tracker.init_tracker(atoms)
        if self.indices:
            self.tracker.filter_indices(self.indices)

    def get_single_trial_move(self) -> List[SystemChange]:
        """
        Create a swap move
        """
        symb_a = np.random.choice(self.tracker.symbols)
        symb_b = np.random.choice([s for s in self.tracker.symbols if s != symb_a])
        rand_pos_a = self.tracker.get_random_indx_of_symbol(symb_a)
        rand_pos_b = self.tracker.get_random_indx_of_symbol(symb_b)
        return [
            SystemChange(index=rand_pos_a, old_symb=symb_a, new_symb=symb_b),
            SystemChange(index=rand_pos_b, old_symb=symb_b, new_symb=symb_a)
        ]

    def on_move_accepted(self, change: Sequence[SystemChange]):
        self.tracker.update_swap_move(change)
