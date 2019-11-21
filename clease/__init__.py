from clease.cleaselogger import CLEASELogger, LogVerbosity
_logger = CLEASELogger()


def set_verbosity(verbosity):
    _logger.verbosity = verbosity


def set_fd(fd):
    _logger.fd = fd

__version__ = '0.9.10'

from clease.template_filters import SkewnessFilter, EquivalentCellsFilter
from clease.template_filters import ValidConcentrationFilter
from clease.template_filters import DistanceBetweenFacetsFilter
from clease.template_filters import VolumeToSurfaceRatioFilter
from clease.template_filters import AngleFilter
from clease.atoms_manager import AtomsManager
from clease.cluster_extractor import ClusterExtractor
from clease.settings_bulk import CEBulk, CECrystal
from clease.evaluate import Evaluate
from clease.corrFunc import CorrFunction
from clease.newStruct import NewStructures
from clease.convexhull import ConvexHull
from clease.concentration import Concentration
from clease.regression import LinearRegression, Tikhonov, Lasso
from clease.ga_fit import GAFit
from clease.bayesian_compressive_sensing import BayesianCompressiveSensing
from clease.cluster_info_mapper import ClusterInfoMapper

__all__ = ['CEBulk', 'CECrystal', 'Concentration', 'CorrFunction',
           'NewStructures', 'NewStructures', 'Evaluate', 'AtomsManager',
           'ConvexHull', 'LinearRegression', 'Tikhonov', 'Lasso',
           'GAFit', 'BayesianCompressiveSensing', 'LogVerbosity',
           'SkewnessFilter', 'ClusterExtractor']
