.. _metadynSampling:

======================
Metadynamics sampling
======================
CLEASE offers the possibility to calculate free energies as a function of
arbitrary collective variables via *metadynamics* sampling. A common collective
variable is the concentration of species, but variable you can think of will
work.
In short we want to calculate curves as shown below

.. figure:: resources/free_energy_illustration.svg
    :width: 60%
    :align: center

Let's move on to the details of how the sampling algorithm works. We start with
the definition of the free energy

.. math:: \exp(-F/kT) = \sum_{\mathbf{\sigma}} \exp(-E(\sigma)/kT) = Z

where *k* is the Boltzmann constant, *T* is the temperature, *E* is the energy
of a configuration, *Z* is the partition function and :math:`\sigma` denotes an
atomic configuration. Hence, the sum runs over all possible configurations.
Furthermore, the probability that the system is in a state :math:`\sigma` is
given by

.. math:: P(\sigma) = \frac{\exp(-E(\sigma)/kT)}{Z}

The free energy at a given value for a general collective variable *q* is
defined by

.. math:: \exp(-F(q)/kT) = \sum_{\mathbf{\sigma}} \delta(f(\sigma) - q)\exp(-E(\sigma)/kT)

the function :math:`f(\sigma)` is a mapping from an atomic configuration to the
sought collective variable. It might for instance return the concentration of
the atomic arrangement. :math:`\delta` is a function that is 1 when
:math:`q = f(\sigma)` and zero otherwise. Thus, the difference is now that
contributions from configurations that has a different value of the collective
variable is cancelled out. Since, we now summed over all configurations that
satisfy :math:`f(\sigma) = q`, the probability of finding the system in any
state that satisfy :math:`f(\sigma) = q` can be obtained by dividing by *Z*

.. math:: P(q) = \frac{\exp(-F(q)/kT)}{Z}

Now, let's see how the probabilities changes if we subtract an artificial
potential *V(q)* that is only a function of the collective variable. First,
we note that this potential can go inside the sum since the sum as only over
configurations that has the same value for *q*. A new free energy *F'* can
therefore be defined as follows

.. math:: \exp(-F'(q)/kT) = \sum_{\mathbf{\sigma}} \delta(f(\sigma) - q)\exp(-(E(\sigma) - V(q))/kT)

by comparison it follows that the relation between the two free energies is

.. math:: F'(q) = F(q) - V(q)

Similarly, the probability of occupying any configuration with
:math:`f(\sigma) = q` in the presence of an artificial potential is

.. math:: P'(q) = \frac{\exp(-F'(q)/kT)}{Z} = \frac{\exp(-(F(q) - V(q))/kT)}{Z}

from the above equation, we note that if we are able to select a potential that
is such that it is exactly equal to the original free energy, the probability
of being in a state satisfying :math:`f(\sigma) = q` is

.. math:: P'(q) = \frac{1}{Z}

which is constant for all values of *q*! Hence, if we partition the domain of
possible *q* values into bins, monitor how often the MC sampler visits each bin
and adaptively tune the artificial potential *V(q)* until we visit all bins
equally often, we know that we have found the free energy.

Carrying out a metadynamics calculation in practice
-----------------------------------------------------
As before, we first need to define the settings. Let's once again use our
favorite example: AuCu!

>>> from clease.settings import CEBulk, Concentration
>>> conc = Concentration(basis_elements=[['Au', 'Cu']])
>>> settings = CEBulk(crystalstructure='fcc',
...                   a=3.8,
...                   supercell_factor=27,
...                   concentration=conc,
...                   db_name="aucu_metadyn.db",
...                   max_cluster_size=2,
...                   max_cluster_dia=[4.0])

The next thing we need to do is to load the ECIs and attach the calculator

>>> eci = {'c0': -1.0, 'c1_0': 0.1, 'c2_d0000_0_00': -0.2}
>>> atoms = settings.atoms.copy()*(5, 5, 5)
>>> from clease.calculator import attach_calculator
>>> atoms = attach_calculator(settings, atoms=atoms, eci=eci)

In pratice, the collective variables are calculated via one of the observers
in CLEASE. If you plan to implement your own observers to use here, please note
that there are certain requirements that needs to be satisfied if an observer
should be applicable for metadynamics calculations.

- The `__call__` method needs to support a `peak` key word. Which is used to
  check what the collective variable is after a move, without actually
  performing the move

>>> def __call__(self, system_changes, peak=False):  # doctest: +SKIP
...    pass

- It needs to have a method *calculate_from_scratch* that takes an atoms object
  as the only argument. This method is used to calculate the collective
  variable from scratch without making use of fast updates when the
  `system_changes` is known

>>> def calculate_from_scratch(self, atoms):  # doctest: +SKIP
...    pass

In this example we are going to use the concentration observer to track the
concentration of Au

>>> from clease.montecarlo.observers import ConcentrationObserver
>>> obs = ConcentrationObserver(atoms, element='Au')

Next, we need to define a sampler. Since, the nature of the problem requires
that the concentration can change, we will use the Semi-Grand Canonical
ensemble

>>> from clease.montecarlo import SGCMonteCarlo
>>> mc = SGCMonteCarlo(atoms, 600, symbols=['Au', 'Cu'])

Then we need to define the artificial bias potential. Here, we are going to use
a binned potential, which is a potential that is defined via values on a grid.

>>> from clease.montecarlo import BinnedBiasPotential
>>> bias = BinnedBiasPotential(xmin=0.0, xmax=1.0, nbins=60, getter=obs)

Here, the minimum concentration is set to 0 and the maximum concentration is
set to 1, and the domain is partitioned into 60 bins. At last, we pass
everything to the metadynamics sampler

>>> from clease.montecarlo import MetaDynamicsSampler
>>> meta_dyn = MetaDynamicsSampler(mc=mc, bias=bias, flat_limit=0.8, mod_factor=0.01,
...                                fname='aucu_metadyn.json')
>>> meta_dyn.run(max_sweeps=1)

The parameter `flat_limit` is a threshold used to determine if we have visited
all the bins equally likely. In the above example, the algorithm will say that
all bins have been visited equally likely if the bins with the fewest visits is
visited at least 80% of the average.

The `mod_factor` tunes how much we should modify the artificial potential when
the sampler visits a bin. It is given in units of *kT*, hence the artifial
potential is altered by *0.01*kT* everytime the sampler visits a bin. Finally,
when we run we set here that the maximum number of sweeps is 1. This is only to
avoid that the trial example takes too long running. This number should be much
higher. If you set it `None`, the algorithm will run until it converges.

When you have managed to converge a calculation, you should reload the previous
estimate, lower the modification factor and run again. Continue to lower the
modification factor until the estimated free energy curve no longer changes.

To load an existing estimate, call this prior to passing the binned potential
to the metadynamics sampler

>>> import json
>>> with open('aucu_metadyn.json', 'r') as f:
...    data = json.load(f)
>>> bias.from_dict(data['bias_pot'])

.. testcleanup::

  import os
  os.remove("aucu_metadyn.json")
  os.remove("aucu_metadyn.db")
