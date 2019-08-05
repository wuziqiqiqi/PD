from clease.cleaselogger import CLEASELogger, LogVerbosity
_logger = CLEASELogger()


def set_verbosity(verbosity):
    _logger.verbosity = verbosity


def set_fd(fd):
    _logger.fd = fd

from clease.settings_bulk import CEBulk, CECrystal
from clease.evaluate import Evaluate
from clease.corrFunc import CorrFunction
from clease.newStruct import NewStructures
from clease.convexhull import ConvexHull
from clease.concentration import Concentration
from clease.regression import LinearRegression, Tikhonov, Lasso
from clease.ga_fit import GAFit
from clease.bayesian_compressive_sensing import BayesianCompressiveSensing

__all__ = ['CEBulk', 'CECrystal', 'Concentration', 'CorrFunction',
           'NewStructures', 'NewStructures', 'Evaluate',
           'ConvexHull', 'LinearRegression', 'Tikhonov', 'Lasso',
           'GAFit', 'BayesianCompressiveSensing', 'LogVerbosity']
