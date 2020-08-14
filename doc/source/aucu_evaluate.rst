.. testsetup::
  :skipif: havedisplay is False

  from clease.settings import Concentration
  from clease import NewStructures
  from clease.settings import CEBulk
  from ase.calculators.emt import EMT
  from ase.db import connect
  from clease.tools import update_db

  conc = Concentration(basis_elements=[['Au', 'Cu']])
  settings = CEBulk(crystalstructure='fcc',
                    a=3.8,
                    supercell_factor=64,
                    concentration=conc,
                    db_name="aucu.db",
                    max_cluster_size=4,
                    max_cluster_dia=[6.0, 4.5, 4.5])

  ns = NewStructures(settings, generation_number=0, struct_per_gen=10)
  ns.generate_initial_pool()

  calc = EMT()
  db = connect("aucu.db")

  for row in db.select(converged=False):
    atoms = row.toatoms()
    atoms.calc = calc
    atoms.get_potential_energy()
    update_db(uid_initial=row.id, final_struct=atoms, db_name="aucu.db")

.. _aucu_evaluate:
.. module:: clease.evaluate

Evaluating the CE model
=======================

We are now ready to evaluate a CE model constructed from the initial 10
calculations. The evaluation of the CE model is performed using :class:`CEBulk`
class, and it supports 3 different linear regression schemes: Bayesian
Compressive Sensing (BCS), :math:`\ell_1` and :math:`\ell_2` regularization.
We will be trying out :math:`\ell_1` and :math:`\ell_2` regularization schemes
to see how they perform using the script below. The script is written to use
:math:`\ell_1` regularization as a fitting scheme (i.e., fitting_scheme='l1'),
and you can change the fitting scheme to :math:`\ell_2` simply by changing it
to 'l2'.

For this tutorial, we use :mod:`EMT <ase.calculators.emt>` calculator to
demonstrate how one can run calculations on the structures generated using
CLEASE and update database with the calculation results for further evaluation
of the CE model. Here is a simple example script that runs the calculations
for all structures that are not yet converged

.. doctest::
  :skipif: havedisplay is False

  >>> from clease import Evaluate
  >>>
  >>> eva = Evaluate(settings=settings, scoring_scheme='k-fold', nsplits=10)
  >>> # scan different values of alpha and return the value of alpha that yields
  >>> # the lowest CV score
  >>> eva.set_fitting_scheme(fitting_scheme='l1')
  >>> alpha = eva.plot_CV(alpha_min=1E-7, alpha_max=1.0, num_alpha=50)
  >>>
  >>> # set the alpha value with the one found above, and fit data using it.
  >>> eva.set_fitting_scheme(fitting_scheme='l1', alpha=alpha)
  >>> eva.plot_fit(interactive=False)
  >>>
  >>> # plot ECI values
  >>> eva.plot_ECI()
  >>>
  >>> # save a dictionary containing cluster names and their ECIs
  >>> eva.save_eci(fname='eci_l1')

.. testcleanup::
  :skipif: havedisplay is False

  import os
  os.remove("aucu.db")
  os.remove("eci_l1.json")

.. autoclass:: Evaluate
   :members: set_fitting_scheme, plot_CV, plot_fit, plot_ECI, save_eci
