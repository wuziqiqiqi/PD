from typing import Dict, Sequence
from abc import ABCMeta
from ase import Atoms
from clease.tools import SystemChange

__all__ = ('BarrierModel', 'SSTEBarrier')


# pylint: disable=too-few-public-methods
class BarrierModel(metaclass=ABCMeta):
    """
    Base barrier model class
    """

    def __call__(self, atoms: Atoms, system_change: Sequence[SystemChange]) -> float:
        raise NotImplementedError("__call__ must be implemented in derived classes")


# pylint: disable=too-few-public-methods
class SSTEBarrier(BarrierModel):
    """
    SSTE is short for Solid Solution Total Energy. It is a simple barrier
    model where the barrier is modelled by the dilute barrier, corrected by
    the change in total energy.

    E_a = Q_s + (E_2 - E_1)/2

    where E_1 is the total energy of starting configuration and E_2 is the
    total energy of the end configuration. Q_s is a constant that depends
    on the species. If an isolated solute is diffusing in a host material
    E_1 = E_2. Thus, Q_s represents the barrier of an isolated solute.

    :param dilute_barrier: Dictionary representing the dilute barrier for
        each species. Example {'Al': 0.05, 'Mg': 0.03}
    """

    def __init__(self, dilute_barrier: Dict[str, float]):
        super().__init__()
        self.dilute_barrier = dilute_barrier

    def __call__(self, atoms: Atoms, system_changes: Sequence[SystemChange]) -> float:
        E1 = atoms.calc.updater.get_energy()
        E2 = atoms.calc.get_energy_given_change(system_changes)

        # Extract the jumping species. The system change should involve
        # a vacancy and another element. Thus, the jumping species is the
        # symbol in system_change that is not a vacancy
        old_symb, new_symb = system_changes[0].old_symb, system_changes[0].new_symb

        # Confirm that one of the symbols is a vacancy
        assert old_symb == 'X' or new_symb == 'X'

        jumping_symb = old_symb if old_symb != 'X' else new_symb
        Ea = self.dilute_barrier[jumping_symb] + 0.5 * (E2 - E1)
        return Ea
