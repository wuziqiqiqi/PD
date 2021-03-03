.. _releasenotes:

=============
Release notes
=============

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
