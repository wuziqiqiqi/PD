.. _releasenotes:

=============
Release notes
=============

Git master branch
=================

* Added release notes

* Added the :mod:`clease.jsonio` module. This has been applied to the
  :class:`clease.settings.ClusterExpansionSettings`, 
  :class:`clease.settings.Concentration` and
  :class:`clease.basis_function.BasisFunction` classes, providing them with
  :func:`save` and :func:`load` functions.

* Tests now automatically run in the pytest temporary directory.
