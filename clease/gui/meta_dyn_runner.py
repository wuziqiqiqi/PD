from clease.calculator import attach_calculator
from clease.montecarlo import SGCMonteCarlo, MetaDynamicsSampler
from clease.gui.meta_dyn_gui_updater import MetaDynGuiUpdater


class MetaDynRunner(object):

    def __init__(self,
                 atoms=None,
                 meta_page=None,
                 max_sweeps=None,
                 settings=None,
                 app_root=None,
                 eci=None,
                 status=None,
                 mc_params=None,
                 T=None,
                 bias=None,
                 flat=None,
                 mod_factor=None,
                 backup_file=None,
                 update_interval=10):
        self.atoms = atoms
        self.T = T
        self.meta_page = meta_page
        self.update_interval = update_interval
        self.max_sweeps = max_sweeps
        self.settings = settings
        self.app_root = app_root
        self.eci = eci
        self.orig_template = settings.atoms.copy()
        self.status = status
        self.mc_params = mc_params
        self.bias = bias
        self.flat = flat
        self.mod_factor = mod_factor
        self.backup_file = backup_file

    def _attach_calc(self):
        self.status.text = 'Attaching calculator...'

        # Temporarily disable info update during initalisation
        self.app_root.info_update_disabled = True
        self.app_root.view_mc_cell_disabled = True
        self.atoms = attach_calculator(settings=self.settings, atoms=self.atoms, eci=self.eci)
        self.app_root.active_template_is_mc_cell = True
        self.app_root.info_update_disabled = False
        self.app_root.view_mc_cell_disabled = False

    def run(self):
        self.meta_page.mc_is_running = True
        update_status_on_end = True
        self._attach_calc()

        try:
            run_calc = True
            self.status.text = 'Running metadynamics MC...'
            if self.mc_params['name'] == 'Semi-grand canonical':
                mc = SGCMonteCarlo(self.atoms, self.T, symbols=self.mc_params['symbols'])
            else:
                msg = f"Unknown MC algorithm. Params: {self.mc_params}"
                self.status.text = msg
                run_calc = False

            if run_calc:
                meta_sampler = MetaDynamicsSampler(mc=mc,
                                                   bias=self.bias,
                                                   mod_factor=self.mod_factor,
                                                   fname=self.backup_file)

                updater = MetaDynGuiUpdater(meta_dyn_page=self.meta_page,
                                            meta_dyn_sampler=meta_sampler)
                meta_sampler.add_observer(updater, interval=self.update_interval)
                self.meta_page.bind_meta_dyn_sampler(meta_sampler)
                meta_sampler.run(max_sweeps=self.max_sweeps)
                self.meta_page.detach_meta_dyn_sampler()
        except Exception as exc:
            self.status.text = str(exc)
            update_status_on_end = False

        # Reset the old template
        self.settings.set_active_template(atoms=self.orig_template)
        self.app_root.active_template_is_mc_cell = False
        self.meta_page.mc_is_running = False

        if update_status_on_end:
            msg = 'MC finished. Lower mod. factor and rerun to improve the '
            msg += 'result'
            self.status.text = msg
