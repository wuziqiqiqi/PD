.. _releasenotes:

=============
Release notes
=============

0.10.3
=======

* Added convex hull plot, :func:`clease.plot_post_process.plot_convex_hull`
* Fixed a bug in :meth:`clease.structgen.NewStructures.generate_gs_structures` where passing multiple atoms objects was failing
* Structure generation of pure elements should now be using the smallest possible cell.
* Alpha and CV values are now stored in the :class:`clease.evaluate.Evaluate` class after running 
  the :meth:`clease.evaluate.Evaluate.alpha_CV` function.
* Added `doc` as an ``extras_require`` in ``setup.py``.
* Other minor bugfixes

0.10.2
=======

* `clease.montecarlo.SSTEBarrier` renamed to `clease.montecarlo.BEPBarrier`

* Added release notes

* Added the :mod:`clease.jsonio` module. This has been applied to the
  :class:`clease.settings.ClusterExpansionSettings`, 
  :class:`clease.settings.Concentration` and
  :class:`clease.basis_function.BasisFunction` classes, providing them with
  :func:`save` and :func:`load` functions.

* Tests now automatically run in the pytest temporary directory.

* Moved ``new_struct`` and ``structure_generator`` into the ``structgen`` module.
  These should now be imported from here, instead.

* Fixed a bug, where the current step counter in the :class:`clease.montecarlo.Montecarlo` class
  would not be reset upon starting a new run.
