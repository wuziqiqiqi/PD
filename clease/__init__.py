# pylint: disable=undefined-variable
import logging
from .version import *

# Do imports in the proper order, avoid cyclic imports
from . import datastructures
from . import basis_function
from . import logging_utils
from . import mp_logger
from . import geometry
from . import cluster
from . import jsonio
from . import tools
from . import db_util
from . import corr_func
from . import settings
from . import calculator
from . import montecarlo
from . import structgen

from .convexhull import *
from .data_manager import *
from .structure_mapper import *
from .structgen.new_struct import NewStructures  # Promotion for convenience
from .svd import *
from .cluster_coverage import *
from .logging_utils import *
from .evaluate import *

# Import this global, so it can be disabled with "clease.REQUIRE_COMPATIBLE_TABLE_VERISON = False"
from .db_util import toggle_require_compatible_table_version

logger = logging.getLogger(__name__)

ADDITIONAL = (
    "__version__",
    "version",
    "settings",
    "basis_function",
    "jsonio",
    "corr_func",
    "structgen",
    "NewStructures",
    "cluster",
    "tools",
    "montecarlo",
    "mp_logger",
    "db_util",
    "calculator",
    "toggle_require_compatible_table_version",
    "logging_utils",
    "datastructures",
    "geometry",
)

__all__ = (
    version.__all__
    + evaluate.__all__
    + convexhull.__all__
    + data_manager.__all__
    + structure_mapper.__all__
    + svd.__all__
    + cluster_coverage.__all__
    + logging_utils.__all__
) + ADDITIONAL
