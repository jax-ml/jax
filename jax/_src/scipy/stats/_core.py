# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from jax import jit
from collections import namedtuple
from jax._src.api import vmap
import jax.numpy as jnp
import warnings
import numpy as np
from jax._src.numpy.lax_numpy import _check_arraylike
from jax._src.numpy.util import _wraps
import scipy
from jax import core
from functools import partial

ModeResult = namedtuple('ModeResult', ('mode', 'count'))

def _contains_nan(a, nan_policy='propagate'):

    policies = ['propagate', 'omit']
    if mode == 'raise':
      a = core.concrete_or_error(jnp.asarray, a,
        "The error occurred because jnp.choose was jit-compiled"
        " with mode='raise'. Use mode='propagate' or mode='omit' instead.")
    elif nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))
    try:
        # Calling jnp.sum to avoid creating a huge array into memory
        # e.g. jnp.isnan(a).any()
        with np.errstate(invalid='ignore'):
            contains_nan = jnp.isnan(jnp.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = jnp.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            nan_policy = 'omit'
            warnings.warn("The input array could not be properly "
                          "checked for nan values. nan values "
                          "will be ignored.", RuntimeWarning)

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return contains_nan, nan_policy

@_wraps(scipy.stats.mode)
@partial(jit, static_argnames=['axis', 'nan_policy'])
def mode(a, axis=0, nan_policy='propagate'):
    _check_arraylike("mode",a)
    if axis is None:
        a = jnp.ravel(a)
        outaxis = 0
    else:
        a = jnp.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = jnp.atleast_1d(a)

    axis = outaxis
    if a.size == 0:
        return ModeResult(jnp.array([]), jnp.array([]))

    contains_nan, nan_policy = _contains_nan(a, nan_policy)

    if contains_nan and nan_policy == 'omit':
        raise NotImplementedError("mode does not support nan values when "
                                  "nan_policy is 'omit'")

    def _mode(x):
        vals, counts = jnp.unique(x, return_counts=True, size=x.size)
        return vals[jnp.argmax(counts)]
    def counts_of_mode(x):
        vals, counts = jnp.unique(x, return_counts=True, size=x.size)
        return counts.max()

    return ModeResult(vmap(_mode, in_axes=(1,))(a.reshape(a.shape[0], -1)).reshape(a.shape[1:]), vmap(counts_of_mode, in_axes=(1,))(a.reshape(a.shape[0], -1)).reshape(a.shape[1:]))
    