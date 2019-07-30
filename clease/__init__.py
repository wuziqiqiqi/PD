from ase.clease.cleaselogger import CLEASELogger, LogVerbosity
_logger = CLEASELogger()

def set_verbosity(verbosity):
    _logger.verbosity = verbosity

def set_fd(fd):
    _logger.fd = fd

from ase.clease.settings_bulk import CEBulk, CECrystal
from ase.clease.evaluate import Evaluate
from ase.clease.corrFunc import CorrFunction
from ase.clease.newStruct import NewStructures
from ase.clease.convexhull import ConvexHull
from ase.clease.concentration import Concentration
from ase.clease.regression import LinearRegression, Tikhonov, Lasso
from ase.clease.ga_fit import GAFit
from ase.clease.bayesian_compressive_sensing import BayesianCompressiveSensing
#from ase.clease.lasso import LASSO

__all__ = ['CEBulk', 'CECrystal', 'Concentration', 'CorrFunction',
           'NewStructures', 'NewStructures', 'Evaluate',
           'ConvexHull', 'LinearRegression', 'Tikhonov', 'Lasso',
           'GAFit', 'BayesianCompressiveSensing', 'LogVerbosity']
