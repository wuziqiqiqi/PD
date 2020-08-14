import numpy as np
from ase.units import kB
import time


class MetaDynGuiUpdater(object):

    def __init__(self, meta_dyn_page=None, meta_dyn_sampler=None):
        self.page = meta_dyn_page
        self.sampler = meta_dyn_sampler

    def __call__(self):
        xmin = self.sampler.bias.xmin
        xmax = self.sampler.bias.xmax
        x = np.linspace(xmin, xmax, 100)
        beta = 1.0 / (kB * self.sampler.mc.T)
        betaG = [-self.sampler.bias.evaluate(y) * beta for y in x]
        self.page.update_free_energy_plot(x, betaG)

        visits = [self.sampler.visit_hist.evaluate(y) for y in x]
        self.page.update_visit_plot(x, visits, self.sampler.flat_limit)
        time.sleep(0.01)
