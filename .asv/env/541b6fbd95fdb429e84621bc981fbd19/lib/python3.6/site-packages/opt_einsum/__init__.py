"""
Main init function for opt_einsum.
"""

from . import blas
from . import helpers
from . import paths
from . import path_random
from .contract import contract, contract_path, contract_expression
from .parser import get_symbol
from .sharing import shared_intermediates
from .paths import BranchBound
from .path_random import RandomGreedy

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions


paths.register_path_fn('random-greedy', path_random.random_greedy)
