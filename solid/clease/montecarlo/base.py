from typing import Union
from abc import ABC
from ase import Atoms
from ase.units import kB
from .mc_evaluator import MCEvaluator, construct_evaluator

__all__ = ("BaseMC",)


class BaseMC(ABC):
    """Base Monte Carlo Class.
    Initializes the internal atoms and evaluator objects.

    Args:
        system (Union[Atoms, MCEvaluator]): Either an ASE Atoms object
            with an attached calculator, or a pre-initialized
            :class:`~clease.montecarlo.mc_evaluator.MCEvaluator`
            object.
        temp (float): Temperature of Monte Carlo simulation in Kelvin
    """

    def __init__(self, system: Union[Atoms, MCEvaluator], temp: float):
        self.evaluator = system
        self.temperature = temp

    @property
    def evaluator(self) -> MCEvaluator:
        """The internal evaluator object.

        :getter: Returns the internal
            :class:`~clease.montecarlo.mc_evaluator.MCEvaluator` object.
        :setter: Sets the internal evaluator object. Can either accept an
            atoms object, or a pre-initialized evaluator object.
            See ``system`` in the docstring of the class constructor.
        :type: :class:`~clease.montecarlo.mc_evaluator.MCEvaluator`
        """
        return self._evaluator

    @evaluator.setter
    def evaluator(self, value: Union[Atoms, MCEvaluator]) -> None:
        """Set the evaluator object. If the value is an Atoms object,
        the evaluator is created on basis of the attached calculator object.
        """
        self._evaluator = construct_evaluator(value)

    @property
    def atoms(self) -> Atoms:
        """The internal Atoms object.

        :type: :class:`ase.atoms.Atoms`
        """
        return self.evaluator.atoms

    def _get_temperature(self) -> float:
        """Retrieve the temperature of the MC."""
        return self._temperature

    def _set_temperature(self, value: float) -> None:
        """Set the internal temperature."""
        self._temperature = value  # pylint: disable=attribute-defined-outside-init
        self.kT = value * kB  # pylint: disable=attribute-defined-outside-init
        self._on_temp_change()

    def _on_temp_change(self) -> None:
        """An MC object may choose to make adjustments after a temperature change,
        by subclassing and adjusting this method."""

    # Ensure that changing the temperature triggers the correct responses.
    temperature = property(
        _get_temperature,
        _set_temperature,
        doc="""
        Property for getting and setting the temperature of the MC object.

        :type: float
        """,
    )
    # For backwards compatibility, alias the "temperature" as "T" as well
    T = property(
        _get_temperature,
        _set_temperature,
        doc="""
                 Alias for the temperature variable.
                 This variable name is deprecated in favor of :py:attr:`~temperature`.

                 :type: float
                 """,
    )
