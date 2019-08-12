from clease.montecarlo import Montecarlo
import numpy as np
from ase.units import kB
from scipy import stats


class InvalidChemicalPotentialError(Exception):
    pass


class SGCMonteCarlo(Montecarlo):
    """
    Class for running Monte Carlo in the Semi-Grand Canonical Ensebmle
    (i.e. fixed number of atoms, but varying composition)

    See docstring of :py:class:`clease.montecarlo.Montecarlo`

    Parameters:

    atoms: Atoms
        Atoms object (with CLEASE calculator attached!)

    temp: float
        Temperature in kelvin

    symbols: list
        Possible symbols to be used in swaps
    """

    def __init__(self, atoms, temp, symbols=[]):
        Montecarlo.__init__(self, atoms, temp)

        self.symbols = symbols

        if len(self.symbols) <= 1:
            raise ValueError("At least 2 symbols have to be specified")
        self.averager = SGCObserver(
            self.atoms.get_calculator(), self, len(self.symbols)-1)

        self.chem_pots = []
        self.chem_pot_names = []
        self.has_attached_avg = False
        self.name = "SGCMonteCarlo"
        self._chemical_potential = None
        self.chem_pot_in_ecis = False
        self.current_singlets = None

        has_attached_obs = False
        for obs in self.observers:
            if obs.name == "SGCObserver":
                has_attached_obs = True
                self.averager = obs
                break
        if not has_attached_obs:
            self.attach(self.averager)

    def _get_trial_move(self):
        """
        Generate a trial move by flipping the symbol of one atom

        :return: Proposed move
        :rtype: List of tuples
        """
        self.current_singlets = self.atoms.get_calculator().get_singlets()
        indx = np.random.randint(low=0, high=len(self.atoms))
        old_symb = self.atoms[indx].symbol
        new_symb = old_symb
        while new_symb == old_symb:
            new_symb = self.symbols[np.random.randint(
                low=0, high=len(self.symbols))]
        system_changes = [(indx, old_symb, new_symb)]
        return system_changes

    def _check_symbols(self):
        """
        Override because there are no restriction on the symbols here
        """
        pass

    def _update_tracker(self, system_changes):
        """
        Override the update of the atom tracker.

        The atom tracker is irrelevant in the semi grand canonical ensemble

        :param list system_changes: Accepted system changes
        """
        pass

    def reset(self):
        """
        Reset the simulation object
        """
        super(SGCMonteCarlo, self).reset()
        self.averager.reset()

    @property
    def chemical_potential(self):
        return self._chemical_potential

    @chemical_potential.setter
    def chemical_potential(self, chem_pot):
        eci = self.atoms.get_calculator().eci
        if any([k not in eci.keys() for k in chem_pot.keys()]):
            raise InvalidChemicalPotentialError(
                "A chemical potential that is currently not tracked is added. Make sure "
                "that all the following keys are in the ECI before the ECI are passed to the "
                "calculator: {} (if not add them with a zero value)"
                "".format(list(chem_pot.keys())))

        self._chemical_potential = chem_pot
        if self.chem_pot_in_ecis:
            self._reset_eci_to_original(self.atoms.get_calculator().eci)
        self._include_chemical_potential_in_ecis(
            chem_pot, self.atoms.get_calculator().eci)

    def _include_chemical_potential_in_ecis(self, chem_potential, eci):
        """
        Including the chemical potentials in the ecis

        :param dict chem_potential: Chemical potentials
        :param dict eci: Original ECIs

        :return: ECIs with chemical potential included
        :rtype: dict
        """
        self.chem_pots = []
        self.chem_pot_names = []
        keys = list(chem_potential.keys())
        keys.sort()
        for key in keys:
            self.chem_pots.append(chem_potential[key])
            self.chem_pot_names.append(key)
            current_eci = eci.get(key, 0.0)
            eci[key] = current_eci - chem_potential[key]
        self.atoms.get_calculator().update_ecis(eci)
        self.chem_pot_in_ecis = True
        self.current_energy = self.atoms.get_calculator().get_energy()
        return eci

    def _reset_eci_to_original(self, eci_with_chem_pot):
        """
        Resets the ecis to their original value

        :parma dict eci_with_chem_pot: ECIs with chemical potential included
        """
        for name, val in zip(self.chem_pot_names, self.chem_pots):
            eci_with_chem_pot[name] += val
        self.atoms.get_calculator().update_ecis(eci_with_chem_pot)
        self.chem_pot_in_ecis = False
        self.current_energy = self.atoms.get_calculator().get_energy()
        return eci_with_chem_pot

    def reset_ecis(self):
        """
        Return the ECIs
        """
        if self.chem_pot_in_ecis:
            self._reset_eci_to_original(self.atoms.get_calculator().eci)

    def run(self, steps=10, chem_potential=None):
        """
        Run Monte Carlo simulation.
        See :py:meth:`cemc.mcmc.Montecarlo.runMC`

        :param dict chem_potential: Chemical potentials.
            The keys should correspond to one of the singlet terms.
            A typical form of this is
            {"c1_0":-1.0,c1_1_1.0}
        """

        if chem_potential is None and self.chemical_potential is None:
            ex_chem_pot = {
                "c1_1": -0.1,
                "c1_2": 0.05
            }
            raise ValueError("No chemicalpotentials given. Has to be "
                             "dictionary of the form {}".format(ex_chem_pot))

        if chem_potential is not None:
            self.chemical_potential = chem_potential
        self.reset()

        mc.Montecarlo.run(self, steps=steps)

    def singlet2composition(self, avg_singlets):
        """Convert singlets to composition."""
        bf = self.atoms.get_calculator().BC.basis_functions
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
            name = s + "_conc"
            res[name] = x[i]
        return res

    def get_thermodynamic(self, reset_ecis=True):
        """
        Compute thermodynamic quantities

        :param bool reset_ecis: If True, the chemical potential will be
            removed from the ECIs

        :return: Thermodynamic quantities
        :rtype: dict
        """
        N = self.averager.counter
        quantities = {}
        singlets = self.averager.singlets/N
        singlets_sq = self.averager.quantities["singlets_sq"]/N

        quantities["sgc_energy"] = self.averager.energy.mean + self.energy_bias
        quantities["sgc_heat_capacity"] = self.averager.energy_sq.mean - \
            self.averager.energy.mean**2

        quantities["sgc_heat_capacity"] /= (kB*self.T**2)

        quantities["energy"] = self.averager.energy.mean + self.energy_bias
        natoms = len(self.atoms)
        for i in range(len(self.chem_pots)):
            quantities["energy"] += self.chem_pots[i]*singlets[i]*natoms

        quantities["temperature"] = self.T
        quantities["n_mc_steps"] = self.averager.counter

        # Add singlets and chemical potential to the dictionary
        for i in range(len(singlets)):
            name = "singlet_{}".format(self.chem_pot_names[i])
            quantities[name] = singlets[i]

            name = "var_singlet_{}".format(self.chem_pot_names[i])
            quantities[name] = singlets_sq[i] - singlets[i]**2

            name = "mu_{}".format(self.chem_pot_names[i])
            quantities[name] = self.chem_pots[i]

        quantities.update(self.meta_info)

        try:
            avg_conc = self.singlet2composition(singlets)
            quantities.update(avg_conc)
        except Exception as exc:
            print("Could not find average singlets!")
            print(exc)

        if reset_ecis:
            self._reset_eci_to_original(self.atoms.get_calculator().eci)
        return quantities