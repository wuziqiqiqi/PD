.. _aucu_probe_gs:


Generating structures for further training
===========================================

You have now seen the initial cross validation (CV) score using 10 initial
training structures. We can further train the CE model using more training
structures to make it more robust.

CLEASE supports 3 ways to generate more strucures. The first (and most
obvious) is generating random structures as you have already done. The
second method is to generate so called "probe structures" which differ the
most from the existing training structures. The third method is to generate
ground-state structures predicted based on current CE model.

Generate probe structures
-------------------------

You can generate probe structures using the following script. Note that it
internally uses simulated annealing algorithm which uses fictitious temperature
values to maximize the difference in correlation function of the new structure.

.. doctest::
  :options: +SKIP

  >>> from clease import NewStructures
  >>> ns = NewStructures(settings, generation_number=1, struct_per_gen=10)
  >>> ns.generate_probe_structure()

Once 10 additional structures are generated, you can re-run the script in
"Running calculations on generated structures" section to calculate their
energies. You should also run the script in "Evaluation of the CE model"
section to evaluate the CV score of the model.It is likely that the CV score
of the model is sufficiently low (few meV/atom or less) at this point.

Generate ground-state structures
--------------------------------

You can now genereate ground-state structures to construct convex-hull
plot of formation energy. The script below generates ground-state
structures with a cell size of :math:`4 \times 4 \times 4` at random
compositions based on current CE model.

.. doctest::
  :options: +SKIP

  >>> from ase.db import connect
  >>> import json
  >>>
  >>> # get template with the cell size = 4x4x4
  >>> template = connect('aucu.db').get(id=17).toatoms()
  >>>
  >>> # import dictionary containing cluster names and their ECIs
  >>> with open('eci_l1.json') as f:
  ...     eci = json.load(f)
  >>>
  >>> ns = NewStructures(settings, generation_number=2, struct_per_gen=10)
  >>>
  >>> ns.generate_gs_structure(atoms=template, init_temp=2000,
  ...                          final_temp=1, num_temp=10,
  ...                          num_steps_per_temp=5000,
  ...                          eci=eci, random_composition=True)


You should re-run the scripts in "Running calculations on generated structures"
and "Evaluating the CE model" sections to see the convex-hull plot and the
latest CV score of the model. If you observe that the CV score
is high (more than ~5 meV/atom), you may want to repeat running the script
for generating ground-state structures.
