from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from ..util import get_module_functions
from .lax_numpy import _not_implemented
from .lax_numpy import IMPLEMENTED_FUNCS

UNIMPLEMENTED_FUNCS = get_module_functions(onp.fft) - set(IMPLEMENTED_FUNCS)
for func in UNIMPLEMENTED_FUNCS:
  if func.__name__ not in globals():
    globals()[func.__name__] = _not_implemented(func)
