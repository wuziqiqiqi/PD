=====================
Monte Carlo Sampling
=====================
Currently CLEASE support two ensembles for Monte Carlo sampling.
First, the canonical ensemble where the number of atoms, concentration 
and temperature is fixed. Secondly, the semi-grand canonical where 
the number of atoms, temperature and chemical potential is fixed.
To use a fitted CE model to run MC sampling we first initialise 
*small* cell holding the nessecary information about the lattice
and the clusters

>>> from clease import Concentration
>>> from clease import CEBulk
>>> conc = Concentration(basis_elements=[['Au', 'Cu']])
>>> setting = CEBulk(crystalstructure='fcc',
...                  a=3.8,
...                  supercell_factor=27,
...                  concentration=conc,
...                  db_name="aucu.db",
...                  max_cluster_size=3,
...                  max_cluster_dia=[6.0, 5.0],
...                  basis_function='polynomial')

Next, we need to specify a set if ECIs. These can for instance be loaded 
from a file, but here we hard code them for simplicity

>>> eci = {'c0': -1.0, 'c2_d0000_0_00': -0.2}

For efficient initialisation of large cells, CLEASE comes with a 
convenient helper function called *attach_calculator*. We create our
MC cell by repeating the *atoms* object of the settings.

>>> atoms = setting.atoms.copy()*(5, 5, 5)
>>> from clease.calculator import attach_calculator
>>> atoms = attach_calculator(setting, atoms=atoms, eci=eci)

Let's insert a few *Cu* atoms

>>> atoms[0].symbol = 'Cu'
>>> atoms[1].symbol = 'Cu'
>>> atoms[2].symbol = 'Cu'

We are now ready to run a MC calculation

>>> from clease.montecarlo import Montecarlo
>>> T = 500
>>> mc = Montecarlo(atoms, T)
>>> mc.run(steps=1000)

After a MC run, you can retrieve internal energy, heat capacity etc. by calling

>>> thermo = mc.get_thermodynamic_quantities()

Monitoring a MC run
====================
In many cases it is useful to be able to monitor the evolution of parameters
during a run, and not simply getting the quantities after the run is finished.
A good example can be to monitor the evolution of the energy in order to determine
whether the system has reached equilibrium. CLEASE comes with a special set of 
classes called *MCObservers* for this task. As an example we can store a value
for the energy every 100 iteration by

>>> from clease.montecarlo.observers import EnergyEvolution
>>> obs = EnergyEvolution(atoms.get_calculator())
>>> mc.attach(obs, interval=100)
>>> mc.run(steps=1000)
>>> energies = obs.energies

Another useful observer is the *Snapshot* observer. This observers
takes snaptshots of the configuration at regular intervals and stores
them in a trajectory file.

>>> from clease.montecarlo.observers import Snapshot
>>> snap = Snapshot(fname='snapshot', atoms=atoms)
>>> mc.attach(snap, interval=200)
>>> mc.run(steps=1000)

There are many more observers distributes with CLEASE, for a complete list
check the API documentation.

Constraining the MC sampling
=============================
In some cases you might want to prevent certain moves to occur.
That can for instance be that certain elements should remain fixed.
CLEASE offers the possibility to impose arbitrary constraint via 
its *MCConstraint* functionality. *MCConstraints* can be added in a
very similar fashion as the observers. To fix one element

>>> from clease.montecarlo.constraints import FixedElement
>>> cnst = FixedElement('Cu')
>>> mc.add_constraint(cnst)

Note, that the usage of a constraint in this system is a bit weird as it 
has only two elements. Hence, fixing one prevents any move from happening.
But the point here is just to illustrate how a constraint can be attached.

Implementing Your Own Observer
===============================
You can implement your own observer and monitor whatever quantity
you might be interested in. To to so you can create your own class that
inherits from the base *MCObserver* class. To illustrate the usage, 
let's create an observers that monitor how many *Cu* atoms there 
are on average in each (100) layer!

Before we initialise this monitor we need to make sure that 
the tag of each atom represents the corresponding layer.

>>> from clease.montecarlo.observers import MCObserver
>>> from ase.geometry import get_layers
>>> class LayerMonitor(MCObserver):
...    def __init__(self, atoms):
...        self.layers, _ = get_layers(atoms, [1, 0, 0])
...        self.layer_average = [0 for _ in set(self.layers)]
...        self.num_calls = 1
...        # Initialise the structure
...        for atom in atoms:
...            if atom.symbol == 'Cu':
...                self.layer_average[self.layers[atom.index]] += 1
...
...    def __call__(self, system_changes):
...        self.num_calls += 1
...        for change in system_changes:
...            layer = self.layers[change[0]]
...            if change[2] == 'Cu':
...                self.layer_average[layer] += 1
...            if change[1] == 'Cu':
...                self.layer_average[layer] -= 1
...
...    def get_averages(self):
...        return {'layer{}'.format(i): x/self.num_calls for i, x in enumerate(self.layer_average)}

When this observer is attached, the `__call__` method will be executed 
on every Monte Carlo step. The `system_changes` parameter is a list of 
the following form `[(10, Au, Cu), (34, Cu, Au)]` which means that the 
symbol on site 10 changes from Au to Cu and the symbol on site 34 changes
from Cu to Au. Hence, in the update algorithm above we check if 
the last element of a single change is equal to Cu, if so we know that 
there is one additional Cu atom in the new layer. And if the middle
element of a change is equal to Cu, there is one less atom in the 
corresponding layer. Note that if a MC move is rejected the `system_changes`
will typically be `[(10, Au, Au), (34, Cu, Cu)]`. The `get_averages` function
returns a dictionary. This method is optinal to implement, but if it is implemented
the result will automatically be added to the result of `get_thermodynamic_quantities`

To use this observer in our calculation

>>> monitor = LayerMonitor(atoms)
>>> mc = Montecarlo(atoms, T)
>>> mc.attach(monitor, interval=1)
>>> mc.run(steps=1000)

There are a few other methods that can be useful to implement. First, 
the `reset` method. This method can be invoked if the `reset` method
of the mc calculation is called.
       