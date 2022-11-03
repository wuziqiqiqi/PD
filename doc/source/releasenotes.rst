.. _releasenotes:

=============
Release notes
=============

1.0.4
======
* Added :class:`~clease.calculator.calc_cache.CleaseCacheCalculator`, as a primitive cache calculator
  object with no cache validation.
* Performance improvements to updating correlation functions.
* Performance improvements to calculating the translation matrix, so the first
  calculation of the clusters should be faster.
* Performance improvements to the :class:`~clease.montecarlo.observers.lowest_energy_observer.LowestEnergyStructure`.
  Correlation functions are also no longer tracked by default, but can be enabled with the ``track_cf`` key.
* The default SGC observer in :class:`~clease.montecarlo.sgc_montecarlo.SGCMonteCarlo` should now be reset
  automatically upon changing the temperature.

1.0.3
======
* Getting thermodynamic quantities in the SGC MC now also retrieves averages from observers.
* Added `interactive` option to :py:func:`~clease.plot_post_process.plot_eci`
* Added :meth:`~clease.settings.ClusterExpansionSettings.get_cluster_corresponding_to_cf_name`.
* Minor performance improvements to SGC MC.
* Added :meth:`~clease.evaluate.Evaluate.set_normalization` for adjusting what elements to normalize by.
  Default is to normalize by everything.

1.0.2
======
* :py:meth:`~clease.structgen.new_struct.NewStructures.insert_structure` returns both
  the initial and final ID if both an initial and final structure was inserted.
* Fixes a bug with writing the Clease calculator to a DB row.

1.0.1
======
* Added the ``ignore_sizes`` keyword to :py:func:`~clease.plot_post_process.plot_eci`
* Changing the maximum cluster diameter will now clear any cached clusters, and
  requires a new build.
* Calling observers in canonical MC can now be disabled with the ``call_observers`` keyword
  for performing burn-in, without executing observers.

1.0.0
======
* 21 June 2022 - CLEASE is no longer considered beta.
* :class:`~clease.evaluate.Evaluate` can now properly support fitting with custom LinearRegression
  schemes, even if they don't support alpha cross-validation.
* :class:`~clease.evaluate.Evaluate` now required explicit calls to
  :py:meth:`~clease.evaluate.Evaluate.fit`. Calls to :py:meth:`~clease.evaluate.Evaluate.get_eci`
  and :py:meth:`~clease.evaluate.Evaluate.get_eci_dict` can no longer implicitly do fitting.
  This un-does a change introduced in version 0.11.6.
* Added the :py:attr:`~clease.montecarlo.montecarlo.Montecarlo.current_accept_rate` property,
  and export the current accept rate in the thermodynamic quantities dictionary under the
  ``accept_rate`` key.
* Removed a series of deprecated things:

  * Removed the ``clease.concentration`` module.
  * Removed the ``clease.new_struct`` module.
  * Removed old regression imports. Regression classes must now be imported from the
    ``clease.regression`` module.
  * Removed the ``clease.structure_generator`` module.
  * Removed the ``max_cluster_size`` settings argument.


0.11.6
=======
* Some small performance optimizations.
* Added a `warn_on_skip` parameter to the
  :py:meth:`~clease.structgen.new_struct.NewStructures.insert_structure` method.
* :class:`~clease.evaluate.Evaluate` should now correctly remember if it doesn't
  need to re-fit the ECI's (see the new :py:meth:`~clease.evaluate.Evaluate.fit` and
  :py:meth:`~clease.evaluate.Evaluate.fit_required` methods).
* Introduced :py:meth:`~clease.evaluate.Evaluate.load_eci` for loading stored ECI values,
  which is convenient for subsequent plotting.
* Added :py:meth:`~clease.montecarlo.kinetic_monte_carlo.KineticMonteCarlo.get_attempt_freq`
  which allows for more flexible customization of the attempt frequencies.
* Added the ``interactive`` keyword to :func:`~clease.plot_post_process.plot_fit`.
* Added an experimental parallelization feature. See :doc:`parallelization`.
* Added ``clease info`` to the CLI to display some information about the installation.

0.11.5
=======
* Fixed a bug with interactive plotting and convex hulls.
* Added the :func:`~clease.geometry.max_sphere_dia_in_cell`
  for calculating sphere diameters within the given cell boundaries.
* Changing the temperature of the :class:`~clease.montecarlo.montecarlo.Montecarlo`
  object will now reset the internal energy averagers.
  Also, :class:`~clease.montecarlo.base.BaseMC` now requires a temperature, and the temperature
  property has been renamed ``temperature``. The old ``T`` attribute name is still accessible
  for backwards compatibility.

  For more information, see `#302 <https://gitlab.com/computationalmaterials/clease/-/issues/302>`_.
* Added :py:meth:`~clease.corr_func.CorrFunction.iter_reconfigure_db_entries`.

0.11.4
=======
* Fixed an issue where :class:`~clease.calculator.util.attach_calculator` would incorrectly
  try to snap the atoms onto a grid.
* Typo in the axis labeling in ``plot_fit``.


0.11.3
=======
* :class:`~clease.datastructures.mc_step.MCStep` and
  :class:`~clease.datastructures.system_change.SystemChange` instances
  are now savable to json via the ``jsonio`` module.
* Fixed a bug which prevented the primitive to have more than 255 atoms.
* The primitive cell is now always wrapped in the settings object.
* Changing ``db_name`` will check if the primitive exists in the new DB,
  and write it if it's missing.
* ``size`` and ``supercell_factor`` are now stored and managed by the
  :class:`~clease.settings.template_atoms.TemplateAtoms` object.
* Made some adjustments to the compilation process.
* Removed the ``include_background_atoms`` setter in the settings object.
  This value must now be set explicitly in the constructor for consistency reasons.
  For more information, see `#292 <https://gitlab.com/computationalmaterials/clease/-/issues/292>`_.
* Fixes a bug with the :class:`~clease.montecarlo.observers.concentration_observer.ConcentrationObserver`.


0.11.2
=======
* Introduces a new :class:`~clease.datastructures.TransMatrix` dataclass
  for the translation matrix.
* Temporarily restricts ASE to ``<3.23``, until we resolve issues with current ASE master.
* Montecarlo will no longer consider background indices in the default swap move generator,
  if background is ignored.
* Added a new :meth:`~clease.montecarlo.montecarlo.Montecarlo.irun` method,
  for iteratively running MC calculations.
* MC observers can now override
  :meth:`~clease.montecarlo.observers.MCObserver.observe_step` instead, which takes a
  :class:`~clease.datastructures.mc_step.MCStep` object.
* Added a new MC observer: :class:`clease.montecarlo.observers.MoveObserver`.

0.11.1
=======
* Fixed a bug in the ``FixedIndices`` constraint class.
* Greatly improved speed of supercell generation - this mostly affects performance
  concerning large supercells.
* Improved performance of the trans matrix generation.

0.11.0
=======
* Python 3.7+ is now required.
* Removed old deprecated functions and classes.
* Some performance improvements.

0.10.9
=======
* Now caches the CF names if requesting every CF name.
  Chops off some of the computation time during a full reconfigure.
* Introduces a new :class:`~clease.data_manager.FinalStructPropertyGetter`, which
  can be used to get arbitrary properties stored as key-value pairs in the database.
  Use the ``prop`` keyword in the :class:`~clease.evaluate.Evaluate` class to use
  this feature.
* Added the ``check_db`` keyword to :class:`~clease.structgen.new_struct.NewStructures`
* Some minor optimizations

0.10.8
=======
* Fixes an issue with the coefficients generated by the Lasso method.
* Fixes an issue with the interactive convex hull plot.
* No longer opens an extra unnecessary GUI window with interactive plots.
* Fixed a bug with the fingerprint grouping, where the relative tolerance would reduce the
  numerical sensitivity too much.
* Now uses the ``packaging`` package for managing version numbers
  and comparisons. Removes usage of the deprecated distutils version comparisons.

0.10.7
=======
* Fixed :meth:`~clease.settings.ClusterExpansionSettings.view_clusters`, which broke in 0.10.6.
* Adds :meth:`~clease.settings.ClusterExpansionSettings.ensure_clusters_exist` and
  :meth:`~clease.settings.ClusterExpansionSettings.get_all_figures_as_atoms`.
  :meth:`~clease.settings.ClusterExpansionSettings.ensure_clusters_exist` can be used to ensure that the
  ``cluster_list`` and ``trans_matrix`` are constructed, but will not cause a reconstruction if they
  are cached.
* Fixed a deprecation warning of ``normalize=True`` in sklearn's ``Lasso`` method.
* Added a benchmarking suite in the tests directory.

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
