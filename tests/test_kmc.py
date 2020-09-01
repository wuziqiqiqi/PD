import os
import pytest
from clease.montecarlo.observers import CorrelationFunctionObserver, EntropyProductionRate
from clease.montecarlo import KineticMonteCarlo, SSTEBarrier, NeighbourSwap
from clease.settings import CEBulk, Concentration
from clease.calculator import attach_calculator


def test_kmc(db_name):
    conc = Concentration(basis_elements=[['Au', 'Cu', 'X']])
    settings = CEBulk(conc,
                      crystalstructure='fcc',
                      size=[1, 1, 1],
                      max_cluster_size=2,
                      max_cluster_dia=[3.0],
                      db_name=db_name)

    eci = {'c0': 0.0, 'c1_0': 0.0, 'c2_d0000_0_00': 0.0}

    atoms = settings.atoms.copy() * (2, 2, 2)
    atoms = attach_calculator(settings, atoms, eci)

    dilute_barriers = {'Au': 0.5, 'Cu': 0.4}

    barrier = SSTEBarrier(dilute_barriers)

    # Insert some Cu
    for i in range(4):
        atoms[i].symbol = 'Cu'
    vac_idx = 5
    atoms[vac_idx].symbol = 'X'

    neighbor = NeighbourSwap(atoms, 3.0)
    for l in neighbor.nl:
        assert len(l) == 12

    T = 300
    kmc = KineticMonteCarlo(atoms, T, barrier, [neighbor])
    obs = CorrelationFunctionObserver(atoms.calc)
    kmc.attach(obs, 2)

    epr_file = 'epr.txt'
    kmc.epr = EntropyProductionRate(buffer_length=2, logfile=epr_file)

    # Check that ValueError is raised if vac_idx is not vacancy
    with pytest.raises(ValueError):
        kmc.run(10, vac_idx - 1)
    kmc.run(10, vac_idx)
    os.remove(epr_file)
