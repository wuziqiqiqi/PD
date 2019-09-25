from clease.calculator import attach_calculator
from clease.montecarlo import Montecarlo
from clease.montecarlo.constraints import ConstrainSwapByBasis
from clease.montecarlo.observers import EnergyEvolution
from clease.montecarlo.observers import EnergyPlotUpdater


class MCRunner(object):
    def __init__(self, atoms=None, eci=None, mc_page=None, conc=None,
                 temps=None, settings=None, sweeps=None, status=None):
        self.atoms = atoms
        self.settings = settings
        self.eci = eci
        self.mc_page = mc_page
        self.conc = conc
        self.conc = conc
        self.temps = temps
        self.num_per_basis = None
        self.basis_elements = settings.concentration.basis_elements
        self.sweeps = sweeps
        self.status = status

    def _attach_calc(self):
        self.status.text = 'Attaching calculator...'
        self.atoms = attach_calculator(setting=self.settings, atoms=self.atoms)

    def _init_conc(self):
        ibb = self.settings.index_by_basis
        for conc_per_basis, indices, symbs in zip(self.conc, ibb,
                                                  self.basis_elements):
            start = 0
            for c, s in zip(conc_per_basis[1:], symbs[1:]):
                num = int(c*len(indices))
                for i in range(start, num+start):
                    self.atoms[indices[i]].symbol = s
                start += num
        formula = self.atoms.get_chemical_formula()
        msg = 'Running MC at fixed conc for {}'.format(formula)
        self.status.text = msg

    def run(self):
        try:
            self.mc_page.mc_is_running = True
            self._attach_calc()
            self._init_conc()

            mc = Montecarlo(self.atoms, 200)
            energy_evol = EnergyEvolution(self.atoms.get_calculator())
            energy_update_rate = 1000
            mc.attach(energy_evol, interval=energy_update_rate)

            energy_plot = EnergyPlotUpdater(
                energy_obs=energy_evol, graph=self.mc_page.energy_graph,
                plot=self.mc_page.energy_plot)
            mc.attach(energy_plot, interval=energy_update_rate)

            cnst = ConstrainSwapByBasis(
                self.atoms, self.settings.index_by_basis)
            mc.add_constraint(cnst)

            for T in self.temps:
                mc.T = T
                self.status.text = 'Current temperature {}K'.format(T)
                mc.run(steps=self.sweeps*len(self.atoms))

        except Exception as exc:
            self.status.text = str(exc)

        self.mc_page.mc_is_running = False
