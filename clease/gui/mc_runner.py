from clease.calculator import attach_calculator
from clease.montecarlo import Montecarlo
from clease.montecarlo.constraints import ConstrainSwapByBasis
from clease.montecarlo.observers import EnergyEvolution
from clease.montecarlo.observers import EnergyPlotUpdater
from clease.tools import wrap_and_sort_by_position
from clease.gui.constants import SYSTEMS_FROM_DB
import traceback
from ase.db import connect


class MCRunner(object):
    """
    Object for running MC on separate thread

    Parameters:
    atoms: Atoms
        Atoms object to use for MC sampling

    eci: dict
        Dictionary with the effective cluster interactions

    conc: list
        List with concentrations per basis. If we have two basis
        with 3 species in one basis and two in the other, an example
        is [(0.5, 0.3, 0.2), (0.2, 0.8)]. It is important that
        the values sum to one in each basis.

    temps: list
        List with temperatures

    settings: ClusterExpansionSettings
        Settings object holder the required information

    sweeps: int
        Number of MC sweeps per temperature

    status: Label
        Status label to output status messages in the GUI

    db_name: str
        Database to be used when storing MC data

    conc_mode: int
        One of SYSTEMS_FROM_DB or CONC_PER_BASIS. In the first case,
        the concentrations are read from an ASE database. In the latter
        the atoms object is initialised randomly such that it has the
        correct concnetration.

    next_mc_obj: MCRunner
        In case of running many MC calculations, MCRunners can be
        chained. If given, the run function of next_mc_obj will be
        launched when this object finished.
    db_id: int
        If conc_mode is SYSTEMS_FROM_DB the concentration is read from
        the item in the database with this ID.
    """

    def __init__(self,
                 atoms=None,
                 eci=None,
                 mc_page=None,
                 conc=None,
                 temps=None,
                 settings=None,
                 sweeps=None,
                 status=None,
                 db_name=None,
                 conc_mode=None,
                 next_mc_obj=None,
                 db_id=None,
                 app_root=None):
        self.atoms = wrap_and_sort_by_position(atoms)
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
        self.orig_template = self.settings.atoms.copy()
        self.db_name = db_name
        self.conc_mode = conc_mode
        self.db_id = db_id
        self.next_mc_obj = next_mc_obj
        self.app_root = app_root

    def _attach_calc(self):
        self.status.text = 'Attaching calculator...'

        # Temporarily disable info update during initalisation
        self.app_root.info_update_disabled = True
        self.app_root.view_mc_cell_disabled = True
        self.atoms = attach_calculator(settings=self.settings, atoms=self.atoms, eci=self.eci)
        self.app_root.active_template_is_mc_cell = True
        self.app_root.info_update_disabled = False
        self.app_root.view_mc_cell_disabled = False

    def _init_conc(self):
        if self.conc_mode == SYSTEMS_FROM_DB:
            self._init_conc_from_db()
            return

        ibb = self.settings.index_by_basis
        for conc_per_basis, indices, symbs in zip(self.conc, ibb, self.basis_elements):
            start = 0
            for c, s in zip(conc_per_basis[1:], symbs[1:]):
                num = int(c * len(indices))
                for i in range(start, num + start):
                    self.atoms[indices[i]].symbol = s
                start += num
        formula = self.atoms.get_chemical_formula()
        msg = f"Running MC at fixed conc for {formula}"
        self.status.text = msg

    def _init_conc_from_db(self):
        db = connect(self.db_name)
        atoms = db.get(id=self.db_id).toatoms()
        atoms = wrap_and_sort_by_position(atoms)

        if len(atoms) != len(self.atoms):
            msg = 'The atoms in the database has to match exactly the '
            msg += ''
            raise ValueError("Currently , ")

        for atom in atoms:
            self.atoms[atom.index].symbol = atom.symbol

    def write_thermodynamic_data_to_db(self, thermo):
        if self.db_name is None:
            return

        if self.db_name == '':
            return

        float_thermo = {}
        for k, v in thermo.items():
            try:
                float_v = float(v)
                float_thermo[k] = float_v
            except Exception:
                pass
        db = connect(self.db_name)

        if self.conc_mode == SYSTEMS_FROM_DB:
            db.update(self.db_id, external_tables={'thermo_data': float_thermo})
        else:
            db.write(self.atoms, external_tables={'thermo_data': float_thermo})

    def run(self):
        try:
            self.mc_page.mc_is_running = True

            self._attach_calc()
            self._init_conc()

            mc = Montecarlo(self.atoms, 200)
            energy_evol = EnergyEvolution(mc, ignore_reset=True)

            energy_update_rate = 5 * len(self.atoms)
            mc.attach(energy_evol, interval=energy_update_rate)

            energy_plot = EnergyPlotUpdater(energy_obs=energy_evol,
                                            graph=self.mc_page.energy_graph,
                                            mean_plot=self.mc_page.mean_energy_plot)
            mc.attach(energy_plot, interval=energy_update_rate)

            cnst = ConstrainSwapByBasis(self.atoms, self.settings.index_by_basis)
            mc.add_constraint(cnst)
            self.mc_page.bind_mc(mc)

            for T in self.temps:
                mc.T = T
                self.status.text = f"Current temperature {T} K"
                mc.run(steps=self.sweeps * len(self.atoms))
                thermo = mc.get_thermodynamic_quantities()
                self.write_thermodynamic_data_to_db(thermo)

            # Reset the old template
            self.settings.set_active_template(atoms=self.orig_template)
            self.app_root.active_template_is_mc_cell = False
            self.mc_page.detach_mc()
            self.status.text = 'MC calculation finished'

            if self.next_mc_obj is not None:
                # Start the next MC runner
                self.next_mc_obj.run()
        except Exception as exc:
            traceback.print_exc()
            self.status.text = str(exc)

        self.mc_page.mc_is_running = False
