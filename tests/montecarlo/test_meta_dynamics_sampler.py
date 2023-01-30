import json
import os
import numpy as np
from ase.build import bulk
from clease.settings import CEBulk, Concentration
from clease.calculator import attach_calculator
from clease.montecarlo import BinnedBiasPotential
from clease.montecarlo import MetaDynamicsSampler
from clease.montecarlo import SGCMonteCarlo
from clease.montecarlo.observers import ConcentrationObserver


def test_ideal_mixture(tmpdir, db_name):
    conc = Concentration(basis_elements=[["Au", "Cu"]])
    settings = CEBulk(conc, a=3.9, db_name=db_name, max_cluster_dia=[3.0], crystalstructure="fcc")
    show_plot = False
    atoms = bulk("Au", a=3.9) * (3, 3, 3)
    eci = {"c0": 0.0, "c1_0": 0.0, "c2_d0000_0_00": 0.0}
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)

    mc = SGCMonteCarlo(atoms, 600, symbols=["Au", "Cu"])
    pot = BinnedBiasPotential(
        xmin=0.0, xmax=1.0, nbins=20, getter=ConcentrationObserver(atoms, element="Au")
    )

    fname = str(tmpdir / "meta_ideal.json")
    meta = MetaDynamicsSampler(mc=mc, bias=pot, fname=fname, mod_factor=0.1)
    meta.log_freq = 0.1
    # NOTE: Mocks use a lot of memory when called many times
    meta.run(max_sweeps=5)
    with open(fname, "r") as infile:
        data = json.load(infile)
    y = np.array(data["betaG"]["y"])
    y -= y[0]

    if show_plot:
        # pylint: disable=import-outside-toplevel
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(data["betaG"]["x"], y / len(atoms), drawstyle="steps")
        x = np.linspace(1e-16, 1 - 1e-16, 100)
        expect = x * np.log(x) + (1 - x) * np.log(1 - x)
        ax.plot(x, -expect)
        plt.show()

    # Check that we have a maximum somerwhere near the center
    # Give quite large tolerance as this is a statistical test
    # the point is to conform that the sampler actually leaves
    # the initial state. For a more rigorous check, run this
    # test manually and set show_plot=True.
    conc_mx = data["betaG"]["x"][np.argmax(y)]
    os.remove(fname)
    assert 0.05 < conc_mx < 0.96


def test_restart_training(db_name, make_tempfile):
    conc = Concentration(basis_elements=[["Au", "Cu"]])
    settings = CEBulk(conc, a=3.9, db_name=db_name, max_cluster_dia=[3.0], crystalstructure="fcc")
    atoms = bulk("Au", a=3.9) * (3, 3, 3)
    eci = {"c0": 0.0, "c1_0": 0.0, "c2_d0000_0_00": 0.0}
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)
    mc = SGCMonteCarlo(atoms, 600, symbols=["Au", "Cu"])
    pot = BinnedBiasPotential(
        xmin=0.0, xmax=1.0, nbins=2, getter=ConcentrationObserver(atoms, element="Au")
    )
    fname = make_tempfile("meta_test_restart.json")
    meta_dyn_data = {
        "bias_pot": {
            "xmin": 0.0,
            "xmax": 1.0,
            "nbins": 2,
            "dx": 0.5,
            "values": [
                1.0,
                1.0,
            ],
        }
    }
    pot.from_dict(meta_dyn_data["bias_pot"])

    meta = MetaDynamicsSampler(mc=mc, bias=pot, fname=fname, mod_factor=1)
    assert meta.mc.current_energy == 1.0, "Biases weren't correctly loaded"
