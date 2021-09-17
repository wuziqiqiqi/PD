.. _releasenotes:

=============
Release notes
=============

0.10.6
=======
* Fixed a bug in the :class:`clease.convexhull.ConvexHull` where multiple end-points wouldn't always find the
  correct minimum energy structure for that end-point.
* Added :class:`~clease.montecarlo.mc_evaluator.MCEvaluator`.
* The settings class should now be much faster to construct, since the construction of the translation matrix and cluster list is defered until requested.
* The built in GUI (based on Kivy) has been removed, in favor of the new Jupyter based `clease-gui <https://clease-gui.readthedocs.io>`_ package.
* Deprecated the use of ``max_cluster_size`` for specifying clusters in :class:`~clease.settings.ClusterExpansionSettings`.
  Clusters should now be specified only though ``max_cluster_dia``,
  where the size of the cluster is infered from the length of the list. The index 0 corresponds to 2-body clusters, index 1 to 3-body etc.,
  i.e. ``max_cluster_dia = [5, 4, 3]`` would result in clusters of up to diameter 5 Å for 2-body clisters, 4 Å for 3-body and
  3 Å for 4-body.


0.10.5
=======
* Added :func:`clease.logging_utils.log_stream` and :func:`clease.logging_utils.log_stream_context` functions to simplify printing the CLEASE logs to a file.
  The global CLEASE logger can be retreived with :func:`clease.logging_utils.get_root_clease_logger`.

0.10.4
=======

* Fixed a bug with sorting the figures in ``ClusterList`` would cause a de-synchronization
  of the indices, and crashing any further usage.
* Now supports clusters of arbitrary size. Used to be limited to 2-, 3- and 4-body clusters.


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
