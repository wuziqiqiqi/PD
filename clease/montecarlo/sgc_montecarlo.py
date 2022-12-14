from typing import Sequence, Dict, Any
import numpy as np
from ase import Atoms
from ase.units import kB
from clease.calculator import Clease
from clease.settings import ClusterExpansionSettings
from .observers import SGCObserver
from .montecarlo import Montecarlo
from .trial_move_generator import TrialMoveGenerator, RandomFlip


class InvalidChemicalPotentialError(Exception):
    pass


class SGCMonteCarlo(Montecarlo):
    """
    Class for running Monte Carlo in the Semi-Grand Canonical Ensebmle
    (i.e., fixed number of atoms, but varying composition)

    See the docstring of :class:`clease.montecarlo.Montecarlo`

    :param atoms: Atoms object (with CLEASE calculator attached!)
    :param temp: Temperature in kelvin
    :param symbols: Possible symbols to be used in swaps
    :param generator: Generator that produces trial moves
    """

    def __init__(
        self,
        atoms: Atoms,
        temp: float,
        symbols: Sequence[str] = (),
        generator: TrialMoveGenerator = None,
        observe_singlets: bool = False,
    ):
        if not isinstance(atoms, Atoms):
            raise ValueError(f"atoms must be an Atoms object, got {atoms!r}")
        if not isinstance(atoms.calc, Clease):
            raise ValueError(
                f"Atoms must have a Clease calculator object attached, got {atoms.calc!r}."
            )

        if generator is None:
            if len(symbols) <= 1:
                raise ValueError("At least 2 symbols have to be specified")
            # Only select indices which are not considered background.
            non_bkg = atoms.calc.settings.non_background_indices
            generator = RandomFlip(symbols, atoms, indices=non_bkg)

        self.averager = SGCObserver(atoms.calc, observe_singlets=observe_singlets)
        super().__init__(atoms, temp, generator=generator)

        self.symbols = symbols

        self.chem_pots = []
        self.chem_pot_names = []
        self.has_attached_avg = False
        self.name = "SGCMonteCarlo"
        self._chemical_potential = None
        self.chem_pot_in_eci = False

        has_attached_obs = False
        for obs in self.iter_observers():
            if obs.name == "SGCObserver":
                has_attached_obs = True
                self.averager = obs
                break
        if not has_attached_obs:
            self.attach(self.averager)

    @property
    def observe_singlets(self) -> bool:
        return self.averager.observe_singlets

    def _check_symbols(self):
        """
        Override because there are no restriction on the symbols here
        """

    def reset(self):
        """
        Reset the simulation object
        """
        super().reset()
        self.averager.reset()

    @property
    def calc(self) -> Clease:
        return self.atoms.calc

    @property
    def settings(self) -> ClusterExpansionSettings:
        return self.calc.settings

    @property
    def chemical_potential(self):
        return self._chemical_potential

    @chemical_potential.setter
    def chemical_potential(self, chem_pot: Dict[str, float]):
        eci = self.calc.eci
        if any(key not in eci for key in chem_pot):
            msg = "A chemical potential not being trackted is added. Make "
            msg += "sure that all the following keys are in the ECIs before "
            msg += "they are passed to the calculator: "
            msg += f"{list(chem_pot.keys())}\n"
            msg += "(Add them with a zero ECI value if they are not supposed "
            msg += "to be included.)"
            raise InvalidChemicalPotentialError(msg)

        self._chemical_potential = chem_pot
        if self.chem_pot_in_eci:
            self._reset_eci_to_original(self.calc.eci)
        self._include_chemical_potential_in_eci(chem_pot, self.calc.eci)

    def _include_chemical_potential_in_eci(self, chem_pot: Dict[str, float], eci: Dict[str, float]):
        """
        Including the chemical potentials in the ECIs

        Parameters:


        chem_pot: dict
            Chemical potentials
        eci: dict
            Original ECIs
        """
        self.chem_pots = []
        self.chem_pot_names = []
        keys = list(chem_pot.keys())
        keys.sort()
        for key in keys:
            self.chem_pots.append(chem_pot[key])
            self.chem_pot_names.append(key)
            current_eci = eci.get(key, 0.0)
            eci[key] = current_eci - chem_pot[key]
        calc = self.calc
        calc.update_eci(eci)
        self.chem_pot_in_eci = True
        self.current_energy = calc.calculate(None, None, None)
        return eci

    def _reset_eci_to_original(self, eci_with_chem_pot: Dict[str, float]):
        """
        Resets the ECIs to their original value

        :parma dict eci_with_chem_pot: ECIs with chemical potential included
        """
        for name, val in zip(self.chem_pot_names, self.chem_pots):
            eci_with_chem_pot[name] += val
        calc = self.calc
        calc.update_eci(eci_with_chem_pot)
        self.chem_pot_in_eci = False
        self.current_energy = calc.calculate(None, None, None)
        return eci_with_chem_pot

    def reset_eci(self):
        """Return the ECIs."""
        if self.chem_pot_in_eci:
            self._reset_eci_to_original(self.calc.eci)

    def run(self, steps: int = 10, call_observers: bool = True, chem_pot: Dict[str, float] = None):
        """
        Run Monte Carlo simulation.
        See :py:meth:`~clease.montecarlo.montecarlo.Montecarlo.run`

        Parameters:

        chem_pot: dict
            Chemical potentials. The keys should correspond to one of the
            singlet terms. A typical form of this is {"c1_0":-1.0,c1_1_1.0}
        """

        if chem_pot is None and self.chemical_potential is None:
            ex_chem_pot = {"c1_1": -0.1, "c1_2": 0.05}
            raise ValueError(
                f"No chemical potentials given. Has to be a dictionary of the form {ex_chem_pot}"
            )

        if chem_pot is not None:
            self.chemical_potential = chem_pot
        self.reset()

        super().run(steps=steps, call_observers=call_observers)

    def singlet2composition(self, avg_singlets: Dict[str, float]):
        """Convert singlets to composition."""
        bf = self.settings.basis_functions
        matrix = np.zeros((len(self.symbols), len(self.symbols)))

        index = {s: i for i, s in enumerate(self.symbols)}
        for i, b in enumerate(bf):
            for s, col in index.items():
                matrix[i, col] = b[s]

        matrix[-1, :] = 1.0
        rhs = np.zeros(len(self.symbols))
        rhs[:-1] = avg_singlets
        rhs[-1] = 1.0
        x = np.linalg.solve(matrix, rhs)

        res = {}
        for s, i in index.items():
            name = f"{s}_conc"
            res[name] = x[i]
        return res

    def reset_averagers(self) -> None:
        """Reset the energy averagers, including the internal SGC Observer"""
        super().reset_averagers()
        # Also remember to reset the internal SGC averager
        self.averager.reset()

    def get_thermodynamic_quantities(self, reset_eci: bool = False) -> Dict[str, Any]:
        """Compute thermodynamic quantities.

        Parameters:

        reset_eci: bool
            If True, the chemical potential will be removed from the ECIs.
        """
        # Note - in order to correctly measure averagers from the SGC observer,
        # we need some information from the SGC MC object. So we directly extract the averages
        # from that observer here.
        avg_obs = self.averager  # Alias
        N = self.averager.counter
        averages = {}
        averages["energy"] = avg_obs.energy.mean
        averages["sgc_energy"] = avg_obs.energy.mean
        averages["sgc_heat_capacity"] = avg_obs.energy_sq.mean - avg_obs.energy.mean**2

        averages["sgc_heat_capacity"] /= kB * self.temperature**2

        averages["temperature"] = self.temperature
        averages["n_mc_steps"] = self.averager.counter
        averages["accept_rate"] = self.current_accept_rate

        # Singlets are more expensive to measure than the other quantities,
        # so only measure them upon request.
        if self.observe_singlets:
            # Add singlets and chemical potential to the dictionary
            # pylint: disable=consider-using-enumerate
            singlets = avg_obs.singlets / N
            singlets_sq = avg_obs.quantities["singlets_sq"] / N

            averages["singlet_energy"] = avg_obs.energy.mean
            natoms = len(self.atoms)
            for i, chem_pot in enumerate(self.chem_pots):
                averages["singlet_energy"] += chem_pot * singlets[i] * natoms
            for i in range(len(singlets)):
                name = f"singlet_{self.chem_pot_names[i]}"
                averages[name] = singlets[i]

                name = f"var_singlet_{self.chem_pot_names[i]}"
                averages[name] = singlets_sq[i] - singlets[i] ** 2

            # Measure concentration from the singlets
            try:
                avg_conc = self.singlet2composition(singlets)
                averages.update(avg_conc)
            # pylint: disable=broad-except
            except Exception as exc:
                print("Could not find average singlets!")
                print(exc)

        # Always measure the chemical potentials.
        for chem_pot_name, chem_pot in zip(self.chem_pot_names, self.chem_pots):
            key = f"mu_{chem_pot_name}"
            averages[key] = chem_pot

        averages.update(self.meta_info)

        # Add information from observers
        averages.update(self._get_obs_averages())

        if reset_eci:
            self._reset_eci_to_original(self.atoms.calc.eci)
        return averages
