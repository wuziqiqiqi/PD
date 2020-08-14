import unittest
from unittest.mock import patch, MagicMock
from clease.montecarlo import BinnedBiasPotential
from clease.montecarlo import MetaDynamicsSampler
from clease.montecarlo import SGCMonteCarlo
from ase.build import bulk
import json
import os
import numpy as np
from clease.montecarlo.observers import ConcentrationObserver


def fake_get_energy_method(self, system_change):
    """
    MC code assumes that the get_energy method updates the
    atoms object
    """
    for change in system_change:
        self.atoms[change[0]].symbol = change[2]
    return 0.0


def test_ideal_mixture(tmpdir):
    show_plot = False
    atoms = bulk('Au') * (3, 3, 3)
    fake_calc = MagicMock()
    fake_calc.get_energy_given_change = MagicMock(
        side_effect=lambda x: fake_get_energy_method(fake_calc, x))
    fake_calc.calculate = MagicMock(return_value=0.0)
    fake_calc.atoms = atoms
    atoms.calc = fake_calc

    mc = SGCMonteCarlo(atoms, 600, symbols=['Au', 'Cu'])
    pot = BinnedBiasPotential(xmin=0.0,
                              xmax=1.0,
                              nbins=20,
                              getter=ConcentrationObserver(atoms, element='Au'))

    fname = str(tmpdir / 'meta_ideal.json')
    meta = MetaDynamicsSampler(mc=mc, bias=pot, fname=fname, mod_factor=0.1)
    meta.log_freq = 0.1
    # NOTE: Mocks use a lot of memory when called many times
    meta.run(max_sweeps=5)
    with open(fname, 'r') as infile:
        data = json.load(infile)
    y = np.array(data['betaG']['y'])
    y -= y[0]

    if show_plot:
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(data['betaG']['x'], y / len(atoms), drawstyle='steps')
        x = np.linspace(1E-16, 1 - 1E-16, 100)
        expect = x * np.log(x) + (1 - x) * np.log(1 - x)
        ax.plot(x, -expect)
        plt.show()

    # Check that we have a maximum somerwhere near the center
    # Give quite large tolerance as this is a statistical test
    # the point is to conform that the sampler actually leaves
    # the initial state. For a more rigorous check, run this
    # test manually and set show_plot=True.
    conc_mx = data['betaG']['x'][np.argmax(y)]
    os.remove(fname)
    assert conc_mx < 0.96 and conc_mx > 0.05
