from typing import Union
from abc import ABC
from ase import Atoms
import numpy as np
import clease
from .mc_evaluator import MCEvaluator, construct_evaluator

__all__ = ('BaseMC',)


class BaseMC(ABC):

    def __init__(
        self,
        system: Union[Atoms, MCEvaluator],
        rng: np.random.Generator = None,
    ):
        """Base Monte Carlo Class.
        Initializes the internal atoms and evaluator objects.

        Args:
            system (Union[Atoms, MCEvaluator]): Either an ASE Atoms object
                with an attached calculator, or a pre-initialized
                :ckass:`~clease.montecarlo.mc_evaluator.MCEvaluator`
                object.
            rng (np.random.Generator, optional): NumPy Generator object to be used
                for the random sampling. If none is specified, the NumPy default_rng
                is used. Defaults to None.
        """
        self.evaluator = system
        # Use the rng object provided by the user, otherwise create a new one.
        self.rng = clease.tools.make_rng_obj(rng=rng)

    @property
    def evaluator(self) -> MCEvaluator:
        return self._evaluator

    @evaluator.setter
    def evaluator(self, value: Union[Atoms, MCEvaluator]) -> None:
        """Set the evaluator object. If the value is an Atoms object,
        the evaluator is created on basis of the attached calculator object.
        """
        self._evaluator = construct_evaluator(value)

    @property
    def atoms(self) -> Atoms:
        return self.evaluator.atoms
