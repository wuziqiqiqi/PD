from __future__ import annotations
from typing import Iterator, List
import ase
from ase.calculators.singlepoint import SinglePointCalculator
from clease.datastructures import MCStep
from .mc_observer import MCObserver

__all__ = ("MoveObserver",)


class MoveObserver(MCObserver):
    """Store each step from an MC run to reconstruct the individual atoms objects later.

    The interval must be set to 1 when attaching this observer,
    as otherwise steps may be lost and the reconstruction may end up being incorrect.

    Args:
        base_atoms (ase.Atoms): The base atoms object which is run in the MC.
        only_accept (bool, optional): Selects whether the only accepted moves or
            all the attempted moves are saved. If False, every move will be saved.
            Defaults to False.
    """

    def __init__(self, base_atoms: ase.Atoms, only_accept: bool = False):
        super().__init__()
        self.base_atoms = base_atoms.copy()
        self.only_accept = only_accept

        self.steps = []

    def observe_step(self, mc_step: MCStep) -> None:
        """Observe a single step"""
        if self.only_accept is True and mc_step.move_accepted is False:
            # Step was rejected, and we only want accepted steps
            return
        self.steps.append(mc_step)

    def reset(self) -> None:
        self.steps = []

    def reconstruct(self) -> List[ase.Atoms]:
        """Rebuild the atoms objects as defined by the observed changes."""
        return list(self.reconstruct_iter())

    def reconstruct_iter(self) -> Iterator[ase.Atoms]:
        """Iterator which builds the atoms objects 1-by-1."""
        atoms = self.base_atoms.copy()
        for step in self.steps:
            # Update the current atoms
            atoms = _apply_step(atoms, step)
            yield atoms

    def interval_ok(self, interval: int) -> bool:
        """Missing steps will result in incorrect reconstructions"""
        return interval == 1


def _apply_step(atoms: ase.Atoms, step: MCStep) -> ase.Atoms:
    """Helper function to apply a step to an atoms object.
    Always a returns a copy of the atoms object."""
    atoms = atoms.copy()
    # Apply changes to a copy of the atoms
    if step.move_accepted:
        for change in step.last_move:
            change.apply_change(atoms)
    # Attach a calculator with the energies
    calc = SinglePointCalculator(atoms, energy=step.energy)
    atoms.calc = calc
    return atoms
