from clease.montecarlo.observers import MCObserver


class EnergyPlotUpdater(MCObserver):
    def __init__(self, energy_obs=None, graph=None, plot=None):
        self.energy_obs = energy_obs
        self.graph = graph
        self.plot = plot

    def __call__(self, system_changes):
        e = self.energy_obs.energy
        self.plot.points = [(i, x - e[0]) for i, x in enumerate(e)]
        xmax = len(e) + 1
        ymin = np.min(e) - e[0]
        ymax = np.max(e) - e[0]
        y_rng = ymax - ymin
        ymin -= 0.05*y_rng
        ymax += 0.05*y_rng
        self.graph.xmax = int(xmax)
        self.graph.ymin = float(ymin)
        self.graph.ymax = float(ymax)
