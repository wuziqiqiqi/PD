from clease.cleaselogger import CLEASELogger, LogVerbosity
_logger = CLEASELogger()


def set_verbosity(verbosity):
    _logger.verbosity = verbosity


def set_fd(fd):
    _logger.fd = fd


__version__ = '0.9.12'

from clease.template_filters import SkewnessFilter, EquivalentCellsFilter
from clease.template_filters import ValidConcentrationFilter
from clease.template_filters import DistanceBetweenFacetsFilter
from clease.template_filters import VolumeToSurfaceRatioFilter
from clease.template_filters import CellVectorDirectionFilter
from clease.template_filters import AngleFilter
from clease.atoms_manager import AtomsManager
from clease.settings import ClusterExpansionSettings
from clease.concentration import Concentration
from clease.settings_bulk import CEBulk, CECrystal, settingsFromJSON
from clease.settings_slab import CESlab
from clease.evaluate import Evaluate
from clease.corrFunc import CorrFunction
from clease.newStruct import NewStructures
from clease.convexhull import ConvexHull
from clease.regression import LinearRegression, Tikhonov, Lasso
from clease.physical_ridge import PhysicalRidge
from clease.ga_fit import GAFit
from clease.bayesian_compressive_sensing import BayesianCompressiveSensing
from clease.sequential_cluster_ridge import SequentialClusterRidge
from clease.cluster_info_mapper import ClusterInfoMapper
from clease.cluster_manager import ClusterManager
from clease.data_manager import (
    DataManager, CorrFuncEnergyDataManager, CorrFuncVolumeDataManager
)

__all__ = ['CEBulk', 'CECrystal', 'Concentration', 'CorrFunction',
           'NewStructures', 'NewStructures', 'Evaluate', 'AtomsManager',
           'ConvexHull', 'LinearRegression', 'Tikhonov', 'Lasso',
           'GAFit', 'BayesianCompressiveSensing', 'LogVerbosity',
           'SkewnessFilter', 'DataManager', 'CorrFuncEnergyDataManager',
           'CorrFuncVolumeDataManager', 'EquivalentCellsFilter',
           'ValidConcentrationFilter', 'DistanceBetweenFacetsFilter',
           'CellVectorDirectionFilter', 'AngleFilter',
           'ClusterExpansionSettings', 'settingsFromJSON', 'CESlab',
           'PhysicalRidge', 'SequentialClusterRidge', 'ClusterInfoMapper',
           'ClusterManager', 'VolumeToSurfaceRatioFilter']
