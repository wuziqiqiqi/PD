from abc import ABC
from ase import Atoms
from clease.datastructures import SystemChanges, MCStep


class MCObserver(ABC):
    """Base class for all MC observers.

    Child observers should override the observe_step() method."""

    name = "GenericObserver"

    def __call__(self, system_changes: SystemChanges) -> None:
        """
        Gets information about the system changes and can perform some action

        :param system_changes: List of system changes. For example, if the
            occupation of the atomic index 23 is changed from Mg to Al,
            system_change = [(23, Mg, Al)]. If an Mg atom occupying the atomic
            index 26 is swapped with an Al atom occupying the atomic index 12,
            system_change = [(26, Mg, Al), (12, Al, Mg)]
        """
        raise NotImplementedError("Observer does not implement __call__")

    def observe_step(self, mc_step: MCStep) -> None:
        """Observe on a MCStep object. Defaults to __call__(system_changes)
        for compatibility reasons. Child classes overriding this function
        should therefore not call the super() version.
        """
        # use the __call__ method for backwards compatibility
        self(mc_step.last_move)

    def reset(self) -> None:
        """Reset all values of the MC observer"""

    def get_averages(self) -> dict:
        """Return averages in the form of a dictionary."""
        # pylint: disable=no-self-use
        return {}

    def calculate_from_scratch(self, atoms: Atoms) -> None:
        """
        Method for calculating the tracked value from scratch
        (i.e. without using fast update methods)
        """

    def interval_ok(self, interval: int) -> bool:
        """
        Check if the interval specified on attach is ok. Default is that all
        intervals are OK

        :param interval: Interval controlling how often a MC observer will be
            called.
        """
        # pylint: disable=no-self-use, unused-argument
        return True
