# pylint: disable=undefined-variable
import logging
from deprecated import deprecated
from .template_filters import *
from .atoms_manager import *
from .convexhull import *
from .data_manager import *
from .structure_mapper import *
from .evaluate import *
from .new_struct import NewStructures  # Promotion for convenience
from .svd import *
from .concentration import *
from .regression_old import *
from .cluster_coverage import *

from . import cluster
from . import new_struct
from . import corr_func
from . import settings
from . import basis_function
from . import montecarlo
from . import mp_logger
from . import tools
from . import template_atoms
from . import concentration

__version__ = '0.10.0'
logger = logging.getLogger(__name__)


@deprecated(version='0.10.0', reason='import CEBulk from clease.settings instead')
def CEBulk(*args, **kwargs):
    return settings.CEBulk(*args, **kwargs)


@deprecated(version='0.10.0', reason='import CECrystal from clease.settings instead')
def CECrystal(*args, **kwargs):
    return settings.CECrystal(*args, **kwargs)


@deprecated(version='0.10.0', reason='import CESlab from clease.settings instead')
def CESlab(*args, **kwargs):
    return settings.CESlab(*args, **kwargs)


ADDITIONAL = ('settings', 'basis_function', 'corr_func', 'new_struct', 'NewStructures', 'cluster',
              'tools', 'montecarlo', 'mp_logger', 'template_atoms', 'CEBulk', 'CECrystal', 'CESlab',
              'concentration')

__all__ = (template_filters.__all__ + atoms_manager.__all__ + concentration.__all__ +
           evaluate.__all__ + convexhull.__all__ + data_manager.__all__ + structure_mapper.__all__ +
           svd.__all__ + regression_old.__all__ + cluster_coverage.__all__ + ADDITIONAL)
