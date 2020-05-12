from clease.montecarlo.observers import MCObserver
from clease.calculator import Clease
from typing import List, Dict, Tuple
from ase.units import kB
import numpy as np


class SGCState(object):
    """
    Represent a thermodynamic state in the semi-grand-canonical ensemble.

    :param temp: Temperature in kelvin
    :param chem_pot: Chemical potentials of the form {c1_0: -0.2, c1_1: 0.3}.
        The function `clease.tools.species_chempot2eci` is useful to convert
        chemical potentials given for each species to chemical potentials
        for each singlet.
    """
    def __init__(self, temp: float, chem_pot: Dict[str, float]):
        self.temp = temp
        self.chem_pot = chem_pot
        self.singlets = {k: 0.0 for k in self.chem_pot.keys()}
        self.avg_weight = 0.0

    def reset(self):
        self.singlets = {k: 0.0 for k in self.chem_pot.keys()}
        self.avg_weight = 0.0

    @property
    def prefix(self) -> str:
        """
        Construct a prefix based on the chemical potentials and the temperature
        """
        prefix = f"{int(self.temp)}K_"
        prefix += '_'.join(f"{k}{sign_indicator(v)}{int(1000.0*abs(v))}"
                           for k, v in self.chem_pot.items())
        return prefix


def sign_indicator(v: float) -> str:
    """
    Return minus if v is negative and plus if v is positive
    """
    if v < 0.0:
        return "minus"
    return "plus"


class MultiStateSGCConcObserver(MCObserver):
    """
    Observer that tracks the concentration at severl different temperatures
    and/or chemical potentials. The observer utilizes the following results.
    Let A be an observable, beta 1/kT, mu the chemical potential and n the
    number of atoms of one of the species in a binary alloy. The average value
    of the observable is given by

            sum_{conf} Aexp(beta*mu*n-beta*E)
    <A> = --------------------------------
                    Z(beta, mu)

    where Z is the partition function. At a different chemical potential mu'
    and inverse temperature beta',

             sum_{conf} Aexp(beta'*mu'*n-beta'*E)
    <A>' = ----------------------------------------
                    Z(beta', mu')

    After some algabraic manipulation one arrives at

              <Aexp((beta'*mu' - beta*mu)*n-(beta-beta')*E)>
    <A>' = ---------------------------------------------------
              <exp((beta'*mu' - beta*mu)*n-(beta-beta')*E)>

    where the averages should be taken at inverse temperature beta and chemical
    potential mu. It should be noted that the predicted value will not be
    accurate if mu' or beta' is very different from the reference values mu
    and beta.

    :param ref_state: Reference state
    :param thermo_states: List of SGCStates where the concentration should
        be tracked
    :param calc: Reference to the calculator attached to the atoms object
        used in the Monte Carlo simulation
    """
    def __init__(self, ref_state: SGCState, thermo_states: List[SGCState],
                 calc: Clease):
        self.thermo_states = thermo_states
        self.name = "multi_state_sgc_conc_observer"
        self.calc = calc
        self.ref_state = ref_state

    def reset(self) -> None:
        """
        Resets the observers to its initial state
        """
        for state in self.thermo_states:
            state.reset()

    def __call__(self, system_changes: List[Tuple[int, str, str]]):
        """
        Observers are called after system update, thus the calcualtor already
        reflects the changes. system_changes is therefore not used.
        """
        beta1 = 1.0/(kB*self.ref_state.temp)
        for state in self.thermo_states:
            cf = self.calc.get_cf()
            beta2 = 1.0/(kB*state.temp)
            betadE = sum((beta2*state.chem_pot[k] -
                          beta1*self.ref_state.chem_pot[k]) *
                         cf[k] for k in state.chem_pot.keys())
            E = self.calc.energy
            weight = np.exp((beta1 - beta2)*E + betadE)
            for k in state.chem_pot.keys():
                state.singlets[k] += weight*cf[k]
                state.avg_weight += weight

    def get_averages(self) -> Dict[str, float]:
        """
        Return a dictionary with the calculated averages
        """
        averages = {}
        for state in self.thermo_states:
            for k, v in state.singlets.items():
                name = state.prefix + f"_singlet_{k}"
                averages[name] = v/state.avg_weight
        return averages
